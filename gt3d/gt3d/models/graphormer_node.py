"""TorchGT's node-level Graphormer (models/graphormer_dist_node_level.py in the TorchGT repo) on gt3d's
attention: pre-LayerNorm transformer blocks, dot-product scores with TorchGT's clamp(-5,5)-exp normalisation
(identical to Exphormer's, no edge gate), attention mask = 1-hop edges of the batch subgraph plus a global
token (node 0) connected to and from every node.  Used for the same-model / same-batch comparison with TorchGT.

Single-GPU version (GraphormerNode) and 2D-distributed version (DistGraphormerNode; LayerNorm is per node, so the
only communication is inside the attention)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gt3d.attention.backend_torch import build_csr
from gt3d.attention.triton_kernel import masked_sparse_attention_triton


class _Block(nn.Module):
    def __init__(self, hidden, ffn, heads, dropout, attention):
        super().__init__()
        self.h, self.dh = heads, hidden // heads
        self.ln1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff1 = nn.Linear(hidden, ffn)
        self.ff2 = nn.Linear(ffn, hidden)
        self.dropout = dropout
        self.attention = attention                     # callable(q, k, v) -> o, all [n, H, dh]

    def forward(self, x):
        n = x.shape[0]
        y = self.ln1(x)
        o = self.attention(self.q(y).view(n, self.h, self.dh), self.k(y).view(n, self.h, self.dh),
                           self.v(y).view(n, self.h, self.dh))
        y = self.o(o.reshape(n, -1))
        x = x + F.dropout(y, self.dropout, self.training)
        y = self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x + F.dropout(y, self.dropout, self.training)


class _FusedDistBlock(_Block):
    """Fused-communication variant of _Block for the 2D grid: the LayerNorm-ed input is gathered ONCE per
    block (row block for Q, column block for K/V, optionally in bf16), Q/K/V are projected locally on the
    gathered rows, the attention returns unnormalised partials and one reduce-scatter over the row group
    lands the normalised output on the owner (same mathematics as DistExphormerLayerFused, no GCN)."""

    def __init__(self, hidden, ffn, heads, dropout, attn_partial, grid, comm_dtype=None, rs_dtype=None):
        super().__init__(hidden, ffn, heads, dropout, attn_partial)
        self.grid, self.comm_dtype, self.rs_dtype = grid, comm_dtype, rs_dtype

    def forward(self, x):
        from gt3d.dist.grid2d import all_gather_rows
        from gt3d.dist.sparse_attention import combine_partials
        g, n = self.grid, x.shape[0]
        y = self.ln1(x)
        cdt = self.comm_dtype or y.dtype
        yc = y.to(cdt)
        y_col = all_gather_rows(yc, g.col_group).to(y.dtype)                # [P_d*chunk, hidden] column block
        y_row = all_gather_rows(yc, g.row_group).to(y.dtype)                # [P_o*chunk, hidden] row block
        q = self.q(y_row).view(-1, self.h, self.dh)
        k = self.k(y_col).view(-1, self.h, self.dh)
        v = self.v(y_col).view(-1, self.h, self.dh)
        acc, l, m = self.attention(q, k, v)
        o, _ = combine_partials(acc, l, m, g, "clampexp", 1e-6, rs_dtype=self.rs_dtype)
        y = self.o(o.reshape(n, -1).to(x.dtype))
        x = x + F.dropout(y, self.dropout, self.training)
        y = self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x + F.dropout(y, self.dropout, self.training)


class GraphormerNode(nn.Module):
    """Single-GPU.  edge_index: [2, E] over the batch nodes where node 0 is the global token (TorchGT layout)."""

    def __init__(self, dim_in, hidden, ffn, heads, layers, num_classes, edge_index, num_nodes,
                 dropout=0.3, input_dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(dim_in, hidden)
        self.graph_token = nn.Embedding(1, hidden)
        self.input_dropout = input_dropout
        crow, col, _, _ = build_csr(edge_index, num_nodes, None)
        self.crow, self.col = crow, col.to(torch.int32)

        def attention(q, k, v):
            return masked_sparse_attention_triton(q, k, v, self.crow, self.col, None, score="dot", norm="clampexp")
        self.layers = nn.ModuleList([_Block(hidden, ffn, heads, dropout, attention) for _ in range(layers)])
        self.final_ln = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = torch.cat([self.graph_token.weight, self.encoder(x)], dim=0)      # global token at index 0
        h = F.dropout(h, self.input_dropout, self.training)
        for layer in self.layers:
            h = layer(h)
        return self.out(self.final_ln(h)[1:])


class DistGraphormerNode(nn.Module):
    """2D-distributed: the owned node chunk holds the global token as node 0 of the padded id space."""

    def __init__(self, grid, attn_block, dim_in, hidden, ffn, heads, layers, num_classes,
                 dropout=0.3, input_dropout=0.1, backend="triton", comm_dtype=None, fused=False, rs_dtype=None):
        super().__init__()
        from gt3d.dist.sparse_attention import DistMaskedSparseAttention, DistAttentionPartial, AttentionBlock
        self.grid = grid
        self.encoder = nn.Linear(dim_in, hidden)
        self.graph_token = nn.Embedding(1, hidden)
        self.input_dropout = input_dropout
        if not isinstance(attn_block, AttentionBlock):
            attn_block = AttentionBlock(grid, *attn_block)                  # build the CSR once for all layers
        if fused:
            self.attns = nn.ModuleList([DistAttentionPartial(grid, attn_block, heads, hidden // heads, backend=backend,
                                                             score="dot", norm="clampexp") for _ in range(layers)])
            self.layers = nn.ModuleList([_FusedDistBlock(hidden, ffn, heads, dropout, a, grid, comm_dtype, rs_dtype)
                                         for a in self.attns])
        else:
            self.attns = nn.ModuleList([DistMaskedSparseAttention(grid, attn_block, heads, hidden // heads, backend=backend,
                                                                  score="dot", norm="clampexp", comm_dtype=comm_dtype)
                                        for _ in range(layers)])
            self.layers = nn.ModuleList([_Block(hidden, ffn, heads, dropout, a) for a in self.attns])
        self.final_ln = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x_loc):
        h = self.encoder(x_loc)
        if self.grid.rank == self.grid.chunk_of(0, 0) or self.grid.owned_range()[0] == 0:
            h = torch.cat([self.graph_token.weight, h[1:]], dim=0)              # global token = global node 0
        h = F.dropout(h, self.input_dropout, self.training)
        for layer in self.layers:
            h = layer(h)
        return self.out(self.final_ln(h))
