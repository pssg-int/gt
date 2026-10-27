"""Exphormer (GCN+Exphormer MultiModel) with a pluggable attention backend.

Same mathematics and parameters as gt3d.models.exphormer_ref.ExphormerRef (so the
official state-dict mapping applies), but the masked sparse attention runs through one
of the backends:
  'ref'    edge-list reference (fp64-friendly, slow)
  'torch'  Backend A, CSR + segment reductions, explicit backward
  'triton' Backend B prototype, fused Triton kernel
The CSR of the attention graph is built once per distinct edge set and cached.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gt3d.attention.ref import masked_sparse_attention_ref
from gt3d.attention.backend_torch import build_csr, masked_sparse_attention_torch
from gt3d.models.exphormer_ref import ExphormerRef, ExphormerLayerRef


class AttentionGraph:
    """Attention edge set (original ∪ expander edges) with a cached CSR."""

    def __init__(self, edge_index, exp_edge_index, num_nodes):
        self.num_nodes = num_nodes
        self.edge_index, self.etype = ExphormerRef.build_attention_edges(edge_index, exp_edge_index)
        crow, col, et, _ = build_csr(self.edge_index, num_nodes, self.etype)
        self.crow, self.col, self.etype_csr = crow, col.to(torch.int32), et.to(torch.int32)


class ExphormerLayer(ExphormerLayerRef):
    def __init__(self, *args, backend="triton", **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend

    def attention(self, x, graph: AttentionGraph, edge_type_emb):
        n, h, d = x.shape[0], self.num_heads, self.d_head
        q = self.q(x).view(n, h, d)
        k = self.k(x).view(n, h, d)
        v = self.v(x).view(n, h, d)
        gate = self.e(edge_type_emb).view(-1, h, d)
        common = dict(gate=gate, score="gated", norm="clampexp", clamp=(-5.0, 5.0), eps=1e-6)
        if self.backend == "ref":
            o = masked_sparse_attention_ref(q, k, v, graph.edge_index, edge_type=graph.etype, **common)
        elif self.backend == "torch":
            o = masked_sparse_attention_torch(q, k, v, graph.crow, graph.col.long(), graph.etype_csr.long(), **common)
        elif self.backend == "triton":
            from gt3d.attention.triton_kernel import masked_sparse_attention_triton
            o = masked_sparse_attention_triton(q, k, v, graph.crow, graph.col, graph.etype_csr, **common)
        else:
            raise ValueError(self.backend)
        return o.reshape(n, h * d)

    def forward(self, x, edge_index, graph: AttentionGraph, edge_type_emb):
        h_local = self.gcn(x, edge_index)
        h_local = F.dropout(h_local, self.dropout, training=self.training)
        h_local = self.bn_local(x + h_local)

        h_attn = self.attention(x, graph, edge_type_emb)
        h_attn = F.dropout(h_attn, self.dropout, training=self.training)
        h_attn = self.bn_attn(x + h_attn)

        h = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(h)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(h + ff)


class Exphormer(ExphormerRef):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, num_heads, dropout=0.3, gnn_dropout=0.3,
                 num_edge_types=2, bn_eps=1e-5, bn_momentum=0.1, backend="triton", pre_l2norm=True):
        super().__init__(dim_in, dim_hidden, dim_out, num_layers, num_heads, dropout, gnn_dropout,
                         num_edge_types, bn_eps, bn_momentum, pre_l2norm=pre_l2norm)
        self.layers = nn.ModuleList([
            ExphormerLayer(dim_hidden, num_heads, dropout, bn_eps, bn_momentum, backend=backend)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, graph: AttentionGraph):
        h = self.pre_mp(x)
        for layer in self.layers:
            h = layer(h, edge_index, graph, self.edge_type_emb.weight)
        return self.head(h)
