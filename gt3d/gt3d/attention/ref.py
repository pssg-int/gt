"""Reference (pure PyTorch, edge-list) implementation of masked sparse attention.

This file is the algebraic specification for the fused kernels and for the
distributed layer.  It is written for clarity and fp64 testing, not for speed:
it materializes per-edge tensors of shape [E, H] and [E, H, d].

Conventions
-----------
* q, k, v have shape [N, H, d] (nodes, heads, head dim).
* edge_index has shape [2, E]; edge_index[0] is the source j (key/value side),
  edge_index[1] is the destination i (query side).  Attention is normalized
  over the in-edges of every destination node i.
* edge_type is an optional [E] long tensor selecting a row of the per-type
  tables `gate` ([T, H, d]) or `bias` ([T, H]).

Score modes
-----------
* 'dot'    : s_ij = scale * <q_i, k_j>                     (UniMP / TransformerConv / Graphormer-node)
* 'gated'  : s_ij = scale * sum_f q_if * k_jf * gate[t_ij, f]   (Exphormer, official code)
* 'biased' : s_ij = scale * <q_i, k_j> + bias[t_ij]        (additive per-type bias)

Normalization modes
-------------------
* 'softmax'  : p_ij = exp(s_ij - max_j s_ij) / sum_j exp(...)      (exact softmax)
* 'clampexp' : p_ij = exp(clamp(s_ij, lo, hi)),  o_i = sum_j p_ij v_j / (sum_j p_ij + eps)
               (this is what the official Exphormer implementation computes; note
               that clamp saturates, so it is NOT equivalent to a softmax)
"""
import math

import torch


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """out[s] = sum over e with index[e] == s of src[e]."""
    out = src.new_zeros((num_segments,) + tuple(src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def segment_max(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """out[s] = max over e with index[e] == s of src[e]; -inf for empty segments."""
    out = torch.full((num_segments,) + tuple(src.shape[1:]), float("-inf"),
                     dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)


def masked_sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_type: torch.Tensor = None,
    gate: torch.Tensor = None,
    bias: torch.Tensor = None,
    score: str = "dot",
    norm: str = "softmax",
    clamp=(-5.0, 5.0),
    eps: float = 1e-6,
    scale: float = None,
    return_stats: bool = False,
):
    """Masked sparse multi-head attention on an explicit edge list.

    Returns o of shape [N, H, d].  With return_stats=True also returns the
    per-row normalizer Z ([N, H]) and, for 'softmax', the row max m ([N, H]).
    """
    assert q.dim() == 3 and k.shape == q.shape and v.shape == q.shape
    n, h, d = q.shape
    src, dst = edge_index[0].long(), edge_index[1].long()
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    prod = q[dst] * k[src] * scale                        # [E, H, d]
    if score == "dot":
        s = prod.sum(-1)                                  # [E, H]
    elif score == "gated":
        assert gate is not None and edge_type is not None
        s = (prod * gate[edge_type]).sum(-1)
    elif score == "biased":
        assert bias is not None and edge_type is not None
        s = prod.sum(-1) + bias[edge_type]
    else:
        raise ValueError(f"unknown score mode {score}")

    vs = v[src]                                           # [E, H, d]
    if norm == "softmax":
        m = segment_max(s, dst, n)                        # [N, H]
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m_safe[dst])
        z = segment_sum(p, dst, n)                        # [N, H]
        o = segment_sum(p.unsqueeze(-1) * vs, dst, n)
        o = o / z.clamp_min(torch.finfo(o.dtype).tiny).unsqueeze(-1)
        stats = (z, m_safe)
    elif norm == "clampexp":
        p = torch.exp(s.clamp(clamp[0], clamp[1]))
        z = segment_sum(p, dst, n)
        o = segment_sum(p.unsqueeze(-1) * vs, dst, n)
        o = o / (z + eps).unsqueeze(-1)
        stats = (z, None)
    else:
        raise ValueError(f"unknown norm mode {norm}")

    if return_stats:
        return o, stats
    return o
