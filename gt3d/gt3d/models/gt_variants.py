"""Two more node-level graph transformers on gt3d's attention operator and 2D layer (generality, E11):

  UniMP (Shi et al. 2021, "Masked Label Prediction"; PyG TransformerConv with gated residual): attention mask = the
     graph's edges + self loops (no expander edges, no global token, no GCN branch, no FFN), dot-product scores,
     softmax; per layer  out = beta * W_r x + (1 - beta) * Attn(x),  beta = sigmoid(W_g [r | o | r - o]), then
     LayerNorm + ReLU; optionally the labels of a random subset of the training nodes are fed as input features.
  Graphormer-node (Ying et al. 2021, the node-level form TorchGT uses): mask = graph edges + self loops + one global
     token connected to and from every node; pre-LayerNorm transformer blocks with dot-product scores + a learnable
     bias per relation type and head (neighbour / self / node->token / token->node: Graphormer's spatial encoding
     restricted to the distances that exist in the sparse mask), softmax, GELU FFN; in-degree (centrality)
     embedding added to the input.

The kernels already provide the three score modes (dot / gated / biased) and both normalisations; what this file
adds is the model wiring: single-process fp64 references (UniMPRef, GraphormerNodeRef) for the parity tests, the
2D-distributed layers (fused-communication form of DistExphormerLayerFused: gather x once per block, project Q/K/V
locally, unnormalised block partials, one reduce-scatter over the row group) and the model wrapper DistGTVariant.
Edge types of the Graphormer mask: 0 neighbour, 1 self loop, 2 node -> token, 3 token -> node.  The global token is
the first padded node id N (every grid pads the id space to a multiple of P * chunk; N_pad > N is required)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from gt3d.dist.grid2d import all_gather_rows
from gt3d.dist.sparse_attention import AttentionBlock, DistAttentionPartial, combine_partials, structure_tensor_bytes

ET_NEIGHBOUR, ET_SELF, ET_TO_TOKEN, ET_FROM_TOKEN = 0, 1, 2, 3


# ------------------------------------------------------------------------------------------------------------
# edge sets
# ------------------------------------------------------------------------------------------------------------
def typed_graph_edges(edge_index, num_nodes):
    """Graph edges + exactly one self loop per node (PyG add_remaining_self_loops: existing loops dropped) with types
    (0 neighbour, 1 self); edge_index [2, E] = (src, dst).  Same edge set as gcn_edges_with_self_loops."""
    dev = edge_index.device
    keep = edge_index[0] != edge_index[1]
    loops = torch.arange(num_nodes, device=dev)
    ei = torch.cat([edge_index[:, keep], torch.stack([loops, loops])], dim=1)
    et = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.long, device=dev),
                    torch.ones(num_nodes, dtype=torch.long, device=dev)])
    return ei, et


def graphormer_edges(edge_index, num_nodes):
    """typed_graph_edges + the global token T = num_nodes: (i -> T) type 2, (T -> i) type 3 for every node i, and the
    token's self loop (type 1).  Node ids run to num_nodes inclusive."""
    dev = edge_index.device
    ei, et = typed_graph_edges(edge_index, num_nodes)
    nodes = torch.arange(num_nodes, device=dev)
    tok = torch.full((num_nodes,), num_nodes, device=dev)
    ei = torch.cat([ei, torch.stack([nodes, tok]), torch.stack([tok, nodes]), torch.tensor([[num_nodes], [num_nodes]], device=dev)], dim=1)
    et = torch.cat([et, torch.full((num_nodes,), ET_TO_TOKEN, device=dev), torch.full((num_nodes,), ET_FROM_TOKEN, device=dev),
                    torch.tensor([ET_SELF], device=dev)])
    return ei, et


def type_local_block(grid, block_edge_index, num_nodes):
    """Edge types of a local (col-local src, row-local dst) block that already contains the self loops: 1 for self
    loops, 0 otherwise."""
    dev = block_edge_index.device
    c_lo, _ = grid.col_block_range()
    row_ids = grid.row_block_ids().to(dev)
    src_g = block_edge_index[0] + c_lo
    dst_g = row_ids[block_edge_index[1]]
    return (src_g == dst_g).long()


def token_edges_local(grid, num_nodes, device):
    """Local (col-local src, row-local dst) edges and types of the global token T = num_nodes for this rank's block,
    generated from the block's own ranges (no global edge list)."""
    T = num_nodes
    assert grid.n_pad > num_nodes, "the global token needs a padded node slot (N_pad > N): change the grid or pad the data"
    c_lo, c_hi = grid.col_block_range()
    row_ids = grid.row_block_ids().to(device)
    real_rows = torch.nonzero(row_ids < num_nodes).squeeze(1)                    # row-local indices of real nodes
    srcs, dsts, types = [], [], []
    tok_in_col = c_lo <= T < c_hi
    r_t, _ = grid.owner_of(torch.tensor([T]))
    tok_in_row = int(r_t[0]) == grid.r
    if tok_in_col:                                                               # token -> every real node of my rows
        t_col = int(grid.global_to_col_local(torch.tensor([T]))[0])
        srcs.append(torch.full_like(real_rows, t_col)); dsts.append(real_rows)
        types.append(torch.full_like(real_rows, ET_FROM_TOKEN))
    if tok_in_row:                                                               # every real node of my columns -> token
        t_row = int(grid.global_to_row_local(torch.tensor([T]))[0])
        cols = torch.arange(0, min(c_hi, num_nodes) - c_lo, device=device)
        srcs.append(cols); dsts.append(torch.full_like(cols, t_row)); types.append(torch.full_like(cols, ET_TO_TOKEN))
        if tok_in_col:
            srcs.append(torch.tensor([t_col], device=device)); dsts.append(torch.tensor([t_row], device=device))
            types.append(torch.tensor([ET_SELF], device=device))
    if not srcs:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return torch.stack([z, z]), z
    return torch.stack([torch.cat(srcs), torch.cat(dsts)]), torch.cat(types)


def owned_in_degree(grid, block_edge_index, block_edge_type, device):
    """In-degree over the neighbour edges (type 0) of the owned nodes: counted on the row block from the local
    block, summed over the row group (every column block contributes its sources)."""
    deg = torch.zeros(grid.n_row_block, dtype=torch.float64, device=device)
    dst = block_edge_index[1][block_edge_type == ET_NEIGHBOUR]
    deg.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float64, device=device))
    dist.all_reduce(deg, group=grid.row_group)
    return deg[grid.c * grid.chunk:(grid.c + 1) * grid.chunk].long()


def token_owned_row(grid, num_nodes):
    """Owned-chunk index of the global token on its owner rank, None elsewhere."""
    lo, hi = grid.owned_range()
    return num_nodes - lo if lo <= num_nodes < hi else None


# ------------------------------------------------------------------------------------------------------------
# single-process references (fp64 parity)
# ------------------------------------------------------------------------------------------------------------
def ref_attention(q, k, v, edge_index, etype=None, bias=None, scale=None):
    """Softmax attention over the edges (src -> dst): q [n, h, d] on the dst side, k / v on the src side."""
    src, dst = edge_index[0], edge_index[1]
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    s = (q[dst] * k[src]).sum(-1) * scale                                        # [E, h]
    if bias is not None:
        s = s + bias[etype]
    m = torch.full((n, h), float("-inf"), dtype=s.dtype, device=s.device)
    m = m.scatter_reduce(0, dst[:, None].expand(-1, h), s.detach(), reduce="amax", include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(s - m[dst])
    l = torch.zeros(n, h, dtype=s.dtype, device=s.device).index_add(0, dst, p)
    alpha = p / l[dst].clamp_min(1e-30)
    return torch.zeros(n, h, d, dtype=q.dtype, device=q.device).index_add(0, dst, alpha.unsqueeze(-1) * v[src])


class UniMPLayerRef(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.h, self.d, self.dropout = heads, dim // heads, dropout
        self.q, self.k, self.v, self.r = (nn.Linear(dim, dim) for _ in range(4))
        self.gate = nn.Linear(3 * dim, 1)
        self.ln = nn.LayerNorm(dim)

    def combine(self, x, o):
        r = self.r(x)
        beta = torch.sigmoid(self.gate(torch.cat([r, o, r - o], dim=-1)))
        out = beta * r + (1.0 - beta) * o
        return F.dropout(F.relu(self.ln(out)), self.dropout, self.training)

    def forward(self, x, edge_index):
        n = x.shape[0]
        o = ref_attention(self.q(x).view(n, self.h, self.d), self.k(x).view(n, self.h, self.d), self.v(x).view(n, self.h, self.d), edge_index)
        return self.combine(x, o.reshape(n, -1))


class UniMPRef(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, num_heads, dropout=0.3, num_classes_label=None):
        super().__init__()
        self.dropout = dropout
        self.pre_lin = nn.Linear(dim_in, dim_hidden)
        self.label_emb = nn.Embedding(num_classes_label + 1, dim_hidden, padding_idx=num_classes_label) if num_classes_label else None
        self.layers = nn.ModuleList([UniMPLayerRef(dim_hidden, num_heads, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, edge_index, label_feat=None):
        h = self.pre_lin(x)
        if self.label_emb is not None and label_feat is not None:
            h = h + self.label_emb(label_feat)
        h = F.dropout(h, self.dropout, self.training)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)


class GraphormerBlockRef(nn.Module):
    def __init__(self, dim, ffn, heads, dropout):
        super().__init__()
        self.h, self.d, self.dropout = heads, dim // heads, dropout
        self.ln1 = nn.LayerNorm(dim)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.ln2 = nn.LayerNorm(dim)
        self.ff1, self.ff2 = nn.Linear(dim, ffn), nn.Linear(ffn, dim)

    def forward(self, x, edge_index, etype, attn_bias):
        n = x.shape[0]
        y = self.ln1(x)
        o = ref_attention(self.q(y).view(n, self.h, self.d), self.k(y).view(n, self.h, self.d), self.v(y).view(n, self.h, self.d),
                          edge_index, etype, attn_bias)
        x = x + F.dropout(self.o(o.reshape(n, -1)), self.dropout, self.training)
        y = self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x + F.dropout(y, self.dropout, self.training)


class GraphormerNodeRef(nn.Module):
    """x has num_nodes + 1 rows (the last one, the token, is ignored); degree [num_nodes + 1]."""

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, num_heads, dropout=0.3, max_degree=64, ffn_mult=2, num_edge_types=4):
        super().__init__()
        self.dropout, self.max_degree = dropout, max_degree
        self.pre_lin = nn.Linear(dim_in, dim_hidden)
        self.degree_emb = nn.Embedding(max_degree + 1, dim_hidden)
        self.graph_token = nn.Embedding(1, dim_hidden)
        self.attn_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads))
        self.layers = nn.ModuleList([GraphormerBlockRef(dim_hidden, ffn_mult * dim_hidden, num_heads, dropout) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(dim_hidden)
        self.head = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, edge_index, etype, degree, token_id):
        h = self.pre_lin(x) + self.degree_emb(degree.clamp(max=self.max_degree))
        h = torch.cat([h[:token_id], self.graph_token.weight, h[token_id + 1:]], dim=0)
        h = F.dropout(h, self.dropout, self.training)
        for layer in self.layers:
            h = layer(h, edge_index, etype, self.attn_bias)
        return self.head(self.final_ln(h))


# ------------------------------------------------------------------------------------------------------------
# 2D-distributed layers (fused communication: x gathered once per block)
# ------------------------------------------------------------------------------------------------------------
def _gather_blocks(x, grid, comm_dtype):
    cdt = comm_dtype or x.dtype
    xc = x.to(cdt)
    x_col = all_gather_rows(xc, grid.col_group).to(x.dtype)                     # [P_d*chunk, dim] column block (K, V)
    x_row = all_gather_rows(xc, grid.row_group).to(x.dtype)                     # [P_o*chunk, dim] row block (Q)
    return x_row, x_col


class DistUniMPLayerFused(UniMPLayerRef):
    def __init__(self, grid, attn_block, dim, heads, dropout, backend="triton", comm_dtype=None, rs_dtype=None):
        super().__init__(dim, heads, dropout)
        self.grid, self.comm_dtype, self.rs_dtype = grid, comm_dtype, rs_dtype
        self.attn = DistAttentionPartial(grid, attn_block, heads, dim // heads, backend=backend, score="dot", norm="softmax")

    def forward(self, x):
        n = x.shape[0]
        x_row, x_col = _gather_blocks(x, self.grid, self.comm_dtype)
        q = self.q(x_row).view(-1, self.h, self.d)
        k = self.k(x_col).view(-1, self.h, self.d)
        v = self.v(x_col).view(-1, self.h, self.d)
        acc, l, m = self.attn(q, k, v)
        o, _ = combine_partials(acc, l, m, self.grid, "softmax", 1e-6, rs_dtype=self.rs_dtype)
        return self.combine(x, o.reshape(n, -1).to(x.dtype))


class DistGraphormerLayerFused(GraphormerBlockRef):
    def __init__(self, grid, attn_block, dim, ffn, heads, dropout, backend="triton", comm_dtype=None, rs_dtype=None):
        super().__init__(dim, ffn, heads, dropout)
        self.grid, self.comm_dtype, self.rs_dtype = grid, comm_dtype, rs_dtype
        self.attn = DistAttentionPartial(grid, attn_block, heads, dim // heads, backend=backend, score="biased", norm="softmax")

    def forward(self, x, attn_bias):
        n = x.shape[0]
        y = self.ln1(x)
        y_row, y_col = _gather_blocks(y, self.grid, self.comm_dtype)
        q = self.q(y_row).view(-1, self.h, self.d)
        k = self.k(y_col).view(-1, self.h, self.d)
        v = self.v(y_col).view(-1, self.h, self.d)
        acc, l, m = self.attn(q, k, v, bias=attn_bias)
        o, _ = combine_partials(acc, l, m, self.grid, "softmax", 1e-6, rs_dtype=self.rs_dtype)
        x = x + F.dropout(self.o(o.reshape(n, -1).to(x.dtype)), self.dropout, self.training)
        y = self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x + F.dropout(y, self.dropout, self.training)


class DistGTVariant(nn.Module):
    """UniMP or Graphormer-node on the 2D grid; parameter names match the references (load_ref_state_dict)."""

    def __init__(self, grid, attn_block_ei, attn_block_et, dim_in, dim_hidden, dim_out, num_layers, num_heads, model="unimp",
                 dropout=0.3, backend="triton", comm_dtype=None, rs_dtype=None, checkpoint=False, num_classes_label=None,
                 max_degree=64, token_row=None, ffn_mult=2):
        super().__init__()
        assert model in ("unimp", "graphormer"), model
        self.grid, self.model, self.dropout, self.checkpoint = grid, model, dropout, checkpoint
        self.max_degree, self.token_row = max_degree, token_row
        block = AttentionBlock(grid, attn_block_ei, attn_block_et)
        self.block = block
        self.pre_lin = nn.Linear(dim_in, dim_hidden)
        self.label_emb = nn.Embedding(num_classes_label + 1, dim_hidden, padding_idx=num_classes_label) if num_classes_label else None
        if model == "unimp":
            self.layers = nn.ModuleList([DistUniMPLayerFused(grid, block, dim_hidden, num_heads, dropout, backend, comm_dtype, rs_dtype)
                                         for _ in range(num_layers)])
        else:
            self.degree_emb = nn.Embedding(max_degree + 1, dim_hidden)
            self.graph_token = nn.Embedding(1, dim_hidden)
            self.attn_bias = nn.Parameter(torch.zeros(4, num_heads))
            self.layers = nn.ModuleList([DistGraphormerLayerFused(grid, block, dim_hidden, ffn_mult * dim_hidden, num_heads, dropout,
                                                                  backend, comm_dtype, rs_dtype) for _ in range(num_layers)])
            self.final_ln = nn.LayerNorm(dim_hidden)
        self.head = nn.Linear(dim_hidden, dim_out)

    def load_ref_state_dict(self, sd):
        self.load_state_dict(sd)

    def forward(self, x_loc, label_feat=None, degree_loc=None):
        h = self.pre_lin(x_loc)
        if self.label_emb is not None and label_feat is not None:
            h = h + self.label_emb(label_feat)
        if self.model == "graphormer":
            if degree_loc is not None:
                h = h + self.degree_emb(degree_loc.clamp(max=self.max_degree))
            if self.token_row is not None:                                           # this rank owns the global token
                t = self.token_row
                h = torch.cat([h[:t], self.graph_token.weight.to(h.dtype), h[t + 1:]], dim=0)
            else:
                h = h + 0.0 * self.graph_token.weight.sum()                            # keep the parameter in the graph (flat all-reduce)
        h = F.dropout(h, self.dropout, self.training)
        for layer in self.layers:
            args = (h, self.attn_bias) if self.model == "graphormer" else (h,)
            if self.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, *args, use_reentrant=False)
            else:
                h = layer(*args)
        if self.model == "graphormer":
            h = self.final_ln(h)
        return self.head(h)

    def structure_bytes(self):
        return structure_tensor_bytes([self.block.crow, self.block.col, self.block.etype], set())

    def compaction_stats(self):
        return 0, 0
