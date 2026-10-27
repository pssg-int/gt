"""Backend A: masked sparse attention on a CSR graph using plain PyTorch ops.

Same semantics as gt3d.attention.ref.masked_sparse_attention_ref, but

* the graph is stored as CSR sorted by destination (crow, col, etype),
* the forward pass uses segment reductions over contiguous rows
  (torch_scatter.segment_csr), never advanced-indexing autograd,
* the backward pass is written explicitly (torch.autograd.Function):
  dQ via segment reductions, dK/dV via index_add_ (atomic adds on GPU),
  so the pathological sort-based `index_put_(accumulate=True)` kernel that
  dominates TorchGT's real-topology backward is never used,
* work is chunked over row ranges so peak memory is O(chunk_edges * H * d)
  instead of O(E * H * d).

This backend is the correctness/safety-net implementation; the fused kernel
(Backend B) replaces it for performance.
"""
import math

import torch
from torch_scatter import segment_csr


def build_csr(edge_index: torch.Tensor, num_nodes: int, edge_type: torch.Tensor = None):
    """Sort edges by destination and return (crow, col, etype_sorted, perm).

    edge_index[0] = source j, edge_index[1] = destination i (same convention as ref.py).
    perm maps CSR edge order -> original edge order.
    """
    src, dst = edge_index[0].long(), edge_index[1].long()
    # canonical order: by destination, then by source.  Sorting the columns inside every row makes the CSR (and the
    # kernels' K/V access pattern) independent of the order the edges arrived in, e.g. of the partition granularity
    # (Reddit at 16 GPUs: 122 ms/epoch from 4x4-block files vs 94 ms from 16x16-block files before this).
    n_cols = int(src.max().item()) + 1 if src.numel() > 0 else 1
    perm = torch.argsort(dst * n_cols + src)
    dst_sorted = dst[perm]
    counts = torch.bincount(dst_sorted, minlength=num_nodes)
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    crow[1:] = torch.cumsum(counts, 0)
    col = src[perm].contiguous()
    etype = edge_type[perm].contiguous() if edge_type is not None else None
    return crow, col, etype, perm


def _row_chunks(crow: torch.Tensor, chunk_edges: int):
    """Split rows into ranges [r0, r1) with roughly <= chunk_edges edges each
    (a single row may exceed the budget; it is kept whole)."""
    n = crow.numel() - 1
    if chunk_edges is None or chunk_edges <= 0 or crow[-1].item() <= chunk_edges:
        return [(0, n)]
    crow_cpu = crow.cpu()
    chunks, r0 = [], 0
    while r0 < n:
        target = crow_cpu[r0].item() + chunk_edges
        r1 = int(torch.searchsorted(crow_cpu, torch.tensor(target), right=True).item()) - 1
        r1 = max(r1, r0 + 1)
        r1 = min(r1, n)
        chunks.append((r0, r1))
        r0 = r1
    return chunks


def _scores(q_rows, k_src, etype, gate, bias, score, scale):
    """q_rows: [E,H,d] (query gathered per edge), k_src: [E,H,d]. Returns s [E,H]
    and the per-edge q*k product needed for the gate gradient (or None)."""
    prod = q_rows * k_src * scale
    if score == "dot":
        return prod.sum(-1), None
    if score == "gated":
        return (prod * gate[etype]).sum(-1), prod
    if score == "biased":
        return prod.sum(-1) + bias[etype], None
    raise ValueError(score)


def torch_attn_fwd_partial(q, k, v, crow, col, etype=None, gate=None, bias=None, *, score="dot",
                           norm="softmax", clamp=(-5.0, 5.0), scale=None, chunk_edges=8_000_000):
    """Unnormalised partials (acc, l, m) of the CSR block; same contract as
    triton_kernel.attn_fwd_partial (m = 0 for clampexp)."""
    lo, hi = float(clamp[0]), float(clamp[1])
    crow = crow.long()                                        # blocks keep int32 pointers; segment_csr needs int64
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    acc = torch.zeros(n, h, d, dtype=q.dtype, device=q.device)
    l = q.new_zeros(n, h)
    m = q.new_zeros(n, h)
    for r0, r1 in _row_chunks(crow, chunk_edges):
        e0, e1 = crow[r0].item(), crow[r1].item()
        if e1 == e0:
            continue
        indptr = crow[r0:r1 + 1] - e0
        rows = torch.repeat_interleave(torch.arange(r0, r1, device=q.device), indptr[1:] - indptr[:-1])
        cols = col[e0:e1]
        et = etype[e0:e1] if etype is not None else None
        s, _ = _scores(q[rows], k[cols], et, gate, bias, score, scale)
        if norm == "softmax":
            mc = segment_csr(s, indptr, reduce="max")
            p = torch.exp(s - mc[rows - r0])
            m[r0:r1] = mc
        else:
            p = torch.exp(s.clamp(lo, hi))
        l[r0:r1] = segment_csr(p, indptr, reduce="sum")
        acc[r0:r1] = segment_csr(p.unsqueeze(-1) * v[cols], indptr, reduce="sum")
    return acc, l, m


def torch_normalize(acc, l, norm, eps=1e-6):
    if norm == "softmax":
        return acc / l.clamp_min(torch.finfo(acc.dtype).tiny).unsqueeze(-1)
    return acc / (l + eps).unsqueeze(-1)


def torch_attn_bwd(q, k, v, crow, col, etype, gate, bias, o, l, m, do, *, score="dot", norm="softmax",
                   clamp=(-5.0, 5.0), eps=1e-6, scale=None, chunk_edges=8_000_000):
    """Backward given the (possibly global) row statistics l, m and normalised output o.
    Returns (dq, dk, dv, dgate, dbias)."""
    lo, hi = float(clamp[0]), float(clamp[1])
    crow = crow.long()                                        # blocks keep int32 pointers; segment_csr needs int64
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    acc_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    dq = torch.zeros_like(q)
    dk = torch.zeros(k.shape, dtype=acc_dtype, device=k.device)
    dv = torch.zeros(v.shape, dtype=acc_dtype, device=v.device)
    dgate = torch.zeros(gate.shape, dtype=acc_dtype, device=gate.device) if (gate is not None and score == "gated") else None
    dbias = torch.zeros(bias.shape, dtype=acc_dtype, device=bias.device) if (bias is not None and score == "biased") else None
    dd = (do * o).sum(-1)                                            # D_i = <dO_i, o_i>  [N,H]
    for r0, r1 in _row_chunks(crow, chunk_edges):
        e0, e1 = crow[r0].item(), crow[r1].item()
        if e1 == e0:
            continue
        indptr = crow[r0:r1 + 1] - e0
        rows = torch.repeat_interleave(torch.arange(r0, r1, device=q.device), indptr[1:] - indptr[:-1])
        cols = col[e0:e1]
        et = etype[e0:e1] if etype is not None else None
        q_rows, k_src, v_src = q[rows], k[cols], v[cols]
        s, prod = _scores(q_rows, k_src, et, gate, bias, score, scale)
        if norm == "softmax":
            p_hat = torch.exp(s - m[rows]) / l[rows].clamp_min(torch.finfo(q.dtype).tiny)
            dp = (do[rows] * v_src).sum(-1) - dd[rows]
            ds = p_hat * dp
        else:
            p = torch.exp(s.clamp(lo, hi))
            inv = 1.0 / (l[rows] + eps)
            p_hat = p * inv
            dp = ((do[rows] * v_src).sum(-1) - dd[rows]) * inv
            ds = p * dp * ((s >= lo) & (s <= hi)).to(p.dtype)        # matches torch.clamp backward
        dv.index_add_(0, cols, (p_hat.unsqueeze(-1) * do[rows]).to(acc_dtype))
        ds_s = (ds * scale).unsqueeze(-1)
        if score == "gated":
            g = gate[et]
            dq[r0:r1] += segment_csr(ds_s * k_src * g, indptr, reduce="sum")
            dk.index_add_(0, cols, (ds_s * q_rows * g).to(acc_dtype))
            dg = ds.unsqueeze(-1) * prod
            for t in range(dgate.shape[0]):                          # T is tiny; masked sums avoid contended atomics
                dgate[t] += dg[et == t].sum(0).to(acc_dtype)
        else:
            dq[r0:r1] += segment_csr(ds_s * k_src, indptr, reduce="sum")
            dk.index_add_(0, cols, (ds_s * q_rows).to(acc_dtype))
            if score == "biased":
                for t in range(dbias.shape[0]):
                    dbias[t] += ds[et == t].sum(0).to(acc_dtype)
    return dq, dk, dv, dgate, dbias


def torch_attn_bwd_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, *, score="dot", norm="softmax",
                           clamp=(-5.0, 5.0), scale=None, chunk_edges=8_000_000):
    """Backward of torch_attn_fwd_partial given gradients (dacc, dl) of the unnormalised partials:
    ds_ij = p_ij (dacc_i . v_j + dl_i) [x clamp mask], dV_j += p_ij dacc_i, p_ij = exp(s_ij - m_i)."""
    lo, hi = float(clamp[0]), float(clamp[1])
    crow = crow.long()                                        # blocks keep int32 pointers; segment_csr needs int64
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    acc_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    dq = torch.zeros_like(q)
    dk = torch.zeros(k.shape, dtype=acc_dtype, device=k.device)
    dv = torch.zeros(v.shape, dtype=acc_dtype, device=v.device)
    dgate = torch.zeros(gate.shape, dtype=acc_dtype, device=gate.device) if (gate is not None and score == "gated") else None
    dbias = torch.zeros(bias.shape, dtype=acc_dtype, device=bias.device) if (bias is not None and score == "biased") else None
    for r0, r1 in _row_chunks(crow, chunk_edges):
        e0, e1 = crow[r0].item(), crow[r1].item()
        if e1 == e0:
            continue
        indptr = crow[r0:r1 + 1] - e0
        rows = torch.repeat_interleave(torch.arange(r0, r1, device=q.device), indptr[1:] - indptr[:-1])
        cols = col[e0:e1]
        et = etype[e0:e1] if etype is not None else None
        q_rows, k_src, v_src = q[rows], k[cols], v[cols]
        s, prod = _scores(q_rows, k_src, et, gate, bias, score, scale)
        if norm == "softmax":
            p = torch.exp(s - m[rows])
            ds = p * ((dacc[rows] * v_src).sum(-1) + dl[rows])
        else:
            p = torch.exp(s.clamp(lo, hi))
            ds = p * ((dacc[rows] * v_src).sum(-1) + dl[rows]) * ((s >= lo) & (s <= hi)).to(p.dtype)
        dv.index_add_(0, cols, (p.unsqueeze(-1) * dacc[rows]).to(acc_dtype))
        ds_s = (ds * scale).unsqueeze(-1)
        if score == "gated":
            g = gate[et]
            dq[r0:r1] += segment_csr(ds_s * k_src * g, indptr, reduce="sum")
            dk.index_add_(0, cols, (ds_s * q_rows * g).to(acc_dtype))
            dg = ds.unsqueeze(-1) * prod
            for t in range(dgate.shape[0]):
                dgate[t] += dg[et == t].sum(0).to(acc_dtype)
        else:
            dq[r0:r1] += segment_csr(ds_s * k_src, indptr, reduce="sum")
            dk.index_add_(0, cols, (ds_s * q_rows).to(acc_dtype))
            if score == "biased":
                for t in range(dbias.shape[0]):
                    dbias[t] += ds[et == t].sum(0).to(acc_dtype)
    return dq, dk, dv, dgate, dbias


class _MaskedSparseAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, crow, col, etype, gate, bias, score, norm, lo, hi, eps, scale, chunk_edges):
        acc, l, m = torch_attn_fwd_partial(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm,
                                           clamp=(lo, hi), scale=scale, chunk_edges=chunk_edges)
        o = torch_normalize(acc, l, norm, eps)
        ctx.save_for_backward(q, k, v, crow, col, etype, gate, bias, o, l, m)
        ctx.cfg = (score, norm, lo, hi, eps, scale, chunk_edges)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, crow, col, etype, gate, bias, o, l, m = ctx.saved_tensors
        score, norm, lo, hi, eps, scale, chunk_edges = ctx.cfg
        dq, dk, dv, dgate, dbias = torch_attn_bwd(q, k, v, crow, col, etype, gate, bias, o, l, m, do,
                                                  score=score, norm=norm, clamp=(lo, hi), eps=eps, scale=scale,
                                                  chunk_edges=chunk_edges)
        dgate = dgate.to(gate.dtype) if dgate is not None else None
        dbias = dbias.to(bias.dtype) if dbias is not None else None
        return (dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, dgate, dbias,
                None, None, None, None, None, None, None)


def masked_sparse_attention_torch(q, k, v, crow, col, etype=None, *, gate=None, bias=None,
                                  score="dot", norm="softmax", clamp=(-5.0, 5.0), eps=1e-6,
                                  scale=None, chunk_edges=8_000_000):
    """Backend A entry point. q,k,v: [N,H,d]; (crow, col, etype) from build_csr."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if score in ("gated", "biased"):
        assert etype is not None
    return _MaskedSparseAttentionTorch.apply(q, k, v, crow, col, etype, gate, bias,
                                            score, norm, float(clamp[0]), float(clamp[1]),
                                            float(eps), float(scale), chunk_edges)
