"""Pure-PyTorch reference re-implementation of the official Exphormer model.

Re-implements graphgps' `MultiModel` with `gt.layer_type: GCN+Exphormer`
(node-level configs such as configs/Exphormer/arxiv.yaml) without any GraphGym
dependency.  It is numerically identical to the official model; see
tests/test_ref_vs_official.py.

Per layer (official MultiLayer with models [LocalModel(GCN), GlobalModel(Exphormer)]):

    h_local = BN( x + Dropout( GCNConv(x, A) ) )
    h_attn  = BN( x + Dropout( ExphormerAttention(x, A ∪ E_exp) ) )
    h       = h_local + h_attn
    h       = BN( h + Dropout( W2 Dropout( ReLU( W1 h ) ) ) )

ExphormerAttention (official semantics, see gt3d/attention/ref.py):
    s_ij = sum_f q_if k_jf g[t_ij, f] / sqrt(d_h),  p_ij = exp(clamp(s_ij, -5, 5)),
    o_i  = sum_j p_ij v_j / (sum_j p_ij + 1e-6),
where t_ij is the edge type (0 = original edge, 1 = expander edge) and
g[t] = E_proj(edge_type_embedding[t]) reshaped to [H, d_h].

Pre-MP (GraphGym GNNPreMP, 1 layer): Linear(no bias) -> BN -> Dropout -> ReLU -> L2 normalize
(cfg.gnn.l2norm defaults to True in GraphGym and arxiv.yaml does not override it).
Head (GraphGym MLP, 1 layer): Linear(bias).
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from gt3d.attention.ref import masked_sparse_attention_ref, segment_sum


def gcn_norm_ref(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
    """Symmetric GCN normalization with PyG's add_remaining_self_loops semantics
    (all existing self loops are dropped and exactly one unit self loop per node
    is added).  Returns (edge_index_with_loops, edge_weight)."""
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    loops = torch.arange(num_nodes, device=edge_index.device)
    src = torch.cat([src[keep], loops])
    dst = torch.cat([dst[keep], loops])
    w = torch.ones(src.numel(), dtype=dtype, device=edge_index.device)
    deg = segment_sum(w, dst, num_nodes)
    dis = deg.pow(-0.5)
    dis = torch.where(torch.isinf(dis), torch.zeros_like(dis), dis)
    w = dis[src] * w * dis[dst]
    return torch.stack([src, dst]), w


class GCNConvRef(nn.Module):
    """PyG GCNConv(dim, dim) with default arguments (add_self_loops=True, normalize=True)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim_out))

    def forward(self, x, edge_index):
        n = x.shape[0]
        ei, w = gcn_norm_ref(edge_index, n, x.dtype)
        h = self.lin(x)
        msg = h[ei[0]] * w.unsqueeze(-1)
        out = segment_sum(msg, ei[1], n)
        return out + self.bias


class ExphormerLayerRef(nn.Module):
    def __init__(self, dim_h, num_heads, dropout, bn_eps, bn_momentum, with_gcn=True):
        super().__init__()
        assert dim_h % num_heads == 0
        self.dim_h, self.num_heads, self.d_head = dim_h, num_heads, dim_h // num_heads
        self.dropout = dropout
        self.with_gcn = with_gcn                 # False: attention-only layer (GraphGPS local_gnn_type=None), no GCN parameters
        if with_gcn:
            # local model (GCN branch)
            self.gcn = GCNConvRef(dim_h, dim_h)
            self.bn_local = nn.BatchNorm1d(dim_h, eps=bn_eps, momentum=bn_momentum)
        # global model (Exphormer attention branch), no output projection in the official code
        self.q = nn.Linear(dim_h, dim_h, bias=False)
        self.k = nn.Linear(dim_h, dim_h, bias=False)
        self.v = nn.Linear(dim_h, dim_h, bias=False)
        self.e = nn.Linear(dim_h, dim_h, bias=False)  # dim_edge == dim_h
        self.bn_attn = nn.BatchNorm1d(dim_h, eps=bn_eps, momentum=bn_momentum)
        # feed-forward block
        self.ff1 = nn.Linear(dim_h, 2 * dim_h)
        self.ff2 = nn.Linear(2 * dim_h, dim_h)
        self.bn_ff = nn.BatchNorm1d(dim_h, eps=bn_eps, momentum=bn_momentum)

    def attention(self, x, attn_edge_index, edge_type, edge_type_emb):
        n, h, d = x.shape[0], self.num_heads, self.d_head
        q = self.q(x).view(n, h, d)
        k = self.k(x).view(n, h, d)
        v = self.v(x).view(n, h, d)
        gate = self.e(edge_type_emb).view(-1, h, d)          # [T, H, d]
        o = masked_sparse_attention_ref(q, k, v, attn_edge_index, edge_type=edge_type,
                                        gate=gate, score="gated", norm="clampexp",
                                        clamp=(-5.0, 5.0), eps=1e-6)
        return o.reshape(n, h * d)

    def forward(self, x, edge_index, attn_edge_index, edge_type, edge_type_emb):
        h_attn = self.attention(x, attn_edge_index, edge_type, edge_type_emb)
        h_attn = F.dropout(h_attn, self.dropout, training=self.training)
        h_attn = self.bn_attn(x + h_attn)
        if not self.with_gcn:
            h = h_attn
        else:
            h_local = self.gcn(x, edge_index)
            h_local = F.dropout(h_local, self.dropout, training=self.training)
            h_local = self.bn_local(x + h_local)
            h = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(h)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(h + ff)


class ExphormerRef(nn.Module):
    """Official Exphormer 'GCN+Exphormer' MultiModel for node classification."""

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, num_heads,
                 dropout=0.3, gnn_dropout=0.3, num_edge_types=2,
                 bn_eps=1e-5, bn_momentum=0.1, pre_l2norm=True, with_gcn=True):
        super().__init__()
        self.gnn_dropout = gnn_dropout
        self.pre_l2norm = pre_l2norm
        # row 0: DummyEdge encoder embedding (original edges); row 1: expander-edge embedding
        self.edge_type_emb = nn.Embedding(num_edge_types, dim_hidden)
        # pre-MP: Linear(no bias) -> BN -> Dropout -> ReLU
        self.pre_lin = nn.Linear(dim_in, dim_hidden, bias=False)
        self.pre_bn = nn.BatchNorm1d(dim_hidden, eps=bn_eps, momentum=bn_momentum)
        self.layers = nn.ModuleList([
            ExphormerLayerRef(dim_hidden, num_heads, dropout, bn_eps, bn_momentum, with_gcn=with_gcn)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim_hidden, dim_out, bias=True)

    @staticmethod
    def build_attention_edges(edge_index, exp_edge_index):
        """Attention edge set = original edges (type 0) ∪ expander edges (type 1)."""
        attn_ei = torch.cat([edge_index, exp_edge_index], dim=1)
        etype = torch.cat([
            torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device),
            torch.ones(exp_edge_index.shape[1], dtype=torch.long, device=edge_index.device),
        ])
        return attn_ei, etype

    def pre_mp(self, x):
        h = self.pre_lin(x)
        h = self.pre_bn(h)
        h = F.dropout(h, self.gnn_dropout, training=self.training)
        h = F.relu(h)
        if self.pre_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h

    def forward(self, x, edge_index, exp_edge_index):
        attn_ei, etype = self.build_attention_edges(edge_index, exp_edge_index)
        h = self.pre_mp(x)
        for layer in self.layers:
            h = layer(h, edge_index, attn_ei, etype, self.edge_type_emb.weight)
        return self.head(h)


# ----------------------------------------------------------------------------
# Mapping from the official graphgps state dict to ExphormerRef
# ----------------------------------------------------------------------------
_BN_KEYS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


def official_to_ref_key(key: str):
    """Map one official MultiModel state-dict key to an ExphormerRef key (or None)."""
    if key == "encoder.edge_encoder.encoder.weight":
        return "edge_type_emb.weight[0]"
    if key == "encoder.exp_edge_fixer.exp_edge_attr.weight":
        return "edge_type_emb.weight[1]"
    if key == "pre_mp.Layer_0.layer.model.weight":
        return "pre_lin.weight"
    m = re.fullmatch(r"pre_mp\.Layer_0\.post_layer\.0\.(\w+)", key)
    if m:
        return f"pre_bn.{m.group(1)}"
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if m:
        i, rest = m.group(1), m.group(2)
        table = {
            "models.0.local_model.lin.weight": "gcn.lin.weight",
            "models.0.local_model.bias": "gcn.bias",
            "models.1.self_attn.Q.weight": "q.weight",
            "models.1.self_attn.K.weight": "k.weight",
            "models.1.self_attn.V.weight": "v.weight",
            "models.1.self_attn.E.weight": "e.weight",
            "ff_linear1.weight": "ff1.weight", "ff_linear1.bias": "ff1.bias",
            "ff_linear2.weight": "ff2.weight", "ff_linear2.bias": "ff2.bias",
        }
        if rest in table:
            return f"layers.{i}.{table[rest]}"
        for prefix, target in (("models.0.norm1_local.", "bn_local."),
                               ("models.1.norm1_attn.", "bn_attn."),
                               ("norm2.", "bn_ff.")):
            if rest.startswith(prefix) and rest[len(prefix):] in _BN_KEYS:
                return f"layers.{i}.{target}{rest[len(prefix):]}"
    m = re.fullmatch(r"post_mp\.layer_post_mp\.model\.0\.model\.(weight|bias)", key)
    if m:
        return f"head.{m.group(1)}"
    return None


@torch.no_grad()
def load_official_state_dict(ref: ExphormerRef, official_sd: dict, strict: bool = True):
    """Copy parameters/buffers of the official graphgps model into `ref`."""
    ref_sd = ref.state_dict()
    used = set()
    for key, val in official_sd.items():
        tgt = official_to_ref_key(key)
        if tgt is None:
            if strict:
                raise KeyError(f"unmapped official key: {key}")
            continue
        m = re.fullmatch(r"(.+)\[(\d+)\]", tgt)
        if m:
            ref_sd[m.group(1)][int(m.group(2))].copy_(val.reshape(-1))
            used.add(m.group(1))
        else:
            ref_sd[tgt].copy_(val)
            used.add(tgt)
    missing = set(ref_sd.keys()) - used
    if strict and missing:
        raise KeyError(f"ref keys not filled: {sorted(missing)}")
    ref.load_state_dict(ref_sd)
    return ref


def ref_to_official_key(ref_key: str, official_keys):
    """Inverse mapping helper for gradient comparison (returns the official key or None)."""
    for k in official_keys:
        if official_to_ref_key(k) == ref_key:
            return k
    return None
