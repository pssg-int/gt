"""Backend B (prototype): fused masked sparse attention in Triton.

Backward = row pass (CSR: dQ, gate/bias grads) + column pass (CSC: dK, dV), both without atomics.

One program per destination row, all heads at once (head-fused gathers of whole
K/V rows).  Forward is a single pass over the row's neighbours with either an
online softmax or Exphormer's clamp-exp normalisation; nothing of size [E, H] is
materialised.  Backward recomputes the attention weights row by row: dQ and the
per-type gate/bias gradients are accumulated in registers, dK/dV are scattered
with atomic adds (the deterministic CSC pass is left for the CUDA kernel).

Semantics are exactly those of gt3d.attention.ref.masked_sparse_attention_ref.
Shapes: q, k, v, o: [N, H, D]; crow: [N+1]; col, etype: [E]; gate: [T, H, D];
bias: [T, H].  H and D are padded to powers of two inside the kernel.
"""
import math
import os

import torch
import triton
import triton.language as tl

_NEG_INF = float("-inf")


def _pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def _block_e(h_pad: int, d_pad: int) -> int:
    return max(4, min(64, 4096 // (h_pad * d_pad)))


def _autotune_configs():
    return [triton.Config({"BLOCK_E": be}, num_warps=nw) for be in (16, 32, 64) for nw in (2, 4, 8)]


@triton.autotune(configs=_autotune_configs(), key=["H", "D", "SCORE", "NORM"])
@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, acc_ptr, l_ptr, m_ptr,
    seg_row_ptr, seg_start_ptr, seg_end_ptr, col_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, eps, SH,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr,
):
    # SH: row stride of q / k / v / acc / gate in heads (= total heads of the tensors; H = heads of this program's
    # head group, a slice of them).  l / m / bias rows have stride SH as well.
    # SCORE: 0 = dot, 1 = gated (per-type diagonal gate), 2 = biased (per-type additive bias)
    # NORM : 0 = online softmax, 1 = clamp-exp (Exphormer)
    # One program per (row, edge segment).  Long rows are split into segments of at most
    # `seg_edges` edges; the un-normalised partial (acc, l, m) of every segment is written to
    # acc_ptr/l_ptr/m_ptr[seg] and combined per row in the wrapper (split-KV style).
    seg = tl.program_id(0)
    row = tl.load(seg_row_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)                       # [H_PAD, D_PAD]
    hd_off = hh[:, None] * D + dd[None, :]
    q = tl.load(q_ptr + row * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([H_PAD, D_PAD], dtype=tl.float32)
    l = tl.zeros([H_PAD], dtype=tl.float32)
    m = tl.full([H_PAD], float("-inf"), dtype=tl.float32)

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        cols = tl.load(col_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]                 # [BLOCK_E, H_PAD, D_PAD]
        goff = cols[:, None, None] * (SH * D) + hd_off[None, :, :]
        kb = tl.load(k_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        prod = q[None, :, :] * kb * scale
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (SH * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)                                   # [BLOCK_E, H_PAD]
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                bmask = emask[:, None] & (hh[None, :] < H)
                s = s + tl.load(bias_ptr + et[:, None] * SH + hh[None, :], mask=bmask, other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        if NORM == 1:
            p = tl.exp(tl.minimum(tl.maximum(s, lo), hi))
            p = tl.where(emask[:, None], p, 0.0)
            l += tl.sum(p, axis=0)
            acc += tl.sum(p[:, :, None] * vb, axis=0)
        else:
            s = tl.where(emask[:, None], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0))
            alpha = tl.exp(m - m_new)                                       # exp(-inf) = 0 on first block
            p = tl.exp(s - m_new[None, :])
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * vb, axis=0)
            m = m_new

    tl.store(acc_ptr + seg * SH * D + hd_off, acc, mask=hd_mask)
    hmask = hh < H
    tl.store(l_ptr + seg * SH + hh, l, mask=hmask)
    tl.store(m_ptr + seg * SH + hh, m, mask=hmask)


@triton.autotune(configs=_autotune_configs(), key=["H", "D", "SCORE", "NORM", "WITH_KV", "PARTIAL"],
                 reset_to_zero=["dgate_ptr", "dbias_ptr", "dk_ptr", "dv_ptr"])   # accumulated with atomics
@triton.jit
def _attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, m_ptr,
    dq_ptr, dk_ptr, dv_ptr, dgate_ptr, dbias_ptr,
    seg_row_ptr, seg_start_ptr, seg_end_ptr, col_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, eps, SH,
    NSLOT: tl.constexpr, T: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, WITH_KV: tl.constexpr,
    PARTIAL: tl.constexpr,
):
    # Row pass (CSR): dQ for the rows of this segment and the per-type gate/bias gradients.
    # With WITH_KV the dK/dV contributions are scattered with atomics (legacy); otherwise they are
    # produced by the column pass _attn_bwd_kv_kernel (deterministic, no atomics).
    # PARTIAL: the upstream gradient is w.r.t. the UNNORMALISED partials (acc = sum p v, l = sum p):
    #   do_ptr holds dacc, o_ptr holds -dl broadcast as D_i (read through the dsum path), l = 1.
    #   Then ds_ij = p_ij (dacc_i . v_j + dl_i), dV_j += p_ij dacc_i  with p_ij = exp(s_ij - m_i)
    #   (m_i the partial's own max; 0 for clamp-exp).
    seg = tl.program_id(0)
    row = tl.load(seg_row_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hmask = hh < H
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)
    hd_off = hh[:, None] * D + dd[None, :]
    q = tl.load(q_ptr + row * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)
    o = tl.load(o_ptr + row * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + row * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)
    l = tl.load(l_ptr + row * SH + hh, mask=hmask, other=1.0).to(tl.float32)
    m = tl.load(m_ptr + row * SH + hh, mask=hmask, other=0.0).to(tl.float32)
    if PARTIAL:
        dsum = tl.load(o_ptr + row * SH + hh, mask=hmask, other=0.0).to(tl.float32)   # -dl_i  [H_PAD]
        inv = tl.zeros([H_PAD], dtype=tl.float32) + 1.0
    else:
        dsum = tl.sum(do * o, axis=1)                                      # D_i  [H_PAD]
        if NORM == 1:
            inv = 1.0 / (l + eps)
        else:
            inv = 1.0 / tl.where(l > 0, l, 1.0)

    dq_acc = tl.zeros([H_PAD, D_PAD], dtype=tl.float32)
    slot = seg % NSLOT

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        cols = tl.load(col_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]
        goff = cols[:, None, None] * (SH * D) + hd_off[None, :, :]
        kb = tl.load(k_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        prod = q[None, :, :] * kb * scale                                   # [BLOCK_E, H_PAD, D_PAD]
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (SH * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                bmask = emask[:, None] & (hh[None, :] < H)
                s = s + tl.load(bias_ptr + et[:, None] * SH + hh[None, :], mask=bmask, other=0.0).to(tl.float32)
        dp = tl.sum(do[None, :, :] * vb, axis=2) - dsum[None, :]           # <dO_i, v_j> - D_i
        if NORM == 1:
            p = tl.exp(tl.minimum(tl.maximum(s, lo), hi))
            p_hat = p * inv[None, :]
            ds = p * dp * inv[None, :]
            ds = tl.where((s >= lo) & (s <= hi), ds, 0.0)
        else:
            p_hat = tl.exp(s - m[None, :]) * inv[None, :]
            ds = p_hat * dp
        ds = tl.where(emask[:, None], ds, 0.0)
        p_hat = tl.where(emask[:, None], p_hat, 0.0)
        ds_s = ds * scale
        if WITH_KV:
            # dV_j += p_hat_ij dO_i ; dK_j += ds_ij * scale * q_i (* gate)
            tl.atomic_add(dv_ptr + goff, p_hat[:, :, None] * do[None, :, :], mask=gmask)
        if SCORE == 1:
            dq_acc += tl.sum(ds_s[:, :, None] * kb * gb, axis=0)
            if WITH_KV:
                tl.atomic_add(dk_ptr + goff, ds_s[:, :, None] * q[None, :, :] * gb, mask=gmask)
            dg = ds[:, :, None] * prod                                       # per-edge gate gradient
            for t in tl.static_range(T):
                sel = tl.where((et == t)[:, None, None], dg, 0.0)
                tl.atomic_add(dgate_ptr + (slot * T + t) * (H * D) + hd_off, tl.sum(sel, axis=0), mask=hd_mask)
        else:
            dq_acc += tl.sum(ds_s[:, :, None] * kb, axis=0)
            if WITH_KV:
                tl.atomic_add(dk_ptr + goff, ds_s[:, :, None] * q[None, :, :], mask=gmask)
            if SCORE == 2:
                for t in tl.static_range(T):
                    sel = tl.where((et == t)[:, None], ds, 0.0)
                    tl.atomic_add(dbias_ptr + (slot * T + t) * H + hh, tl.sum(sel, axis=0), mask=hmask)

    tl.store(dq_ptr + seg * SH * D + hd_off, dq_acc, mask=hd_mask)


@triton.autotune(configs=_autotune_configs(), key=["H", "D", "SCORE", "NORM"])
@triton.jit
def _attn_bwd_kv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, m_ptr, dsum_ptr,
    dk_ptr, dv_ptr,
    seg_col_ptr, seg_start_ptr, seg_end_ptr, rowidx_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, eps, SH,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr,
):
    # Column pass (CSC): one program per (source column j, segment of its out-edges).  Recomputes
    # p_ij from the global row statistics (l_i, m_i) and D_i = <dO_i, o_i>, and accumulates
    #   dK_j = sum_i ds_ij * scale * q_i (* g_t),   dV_j = sum_i p_hat_ij dO_i
    # in registers; partials of split columns are combined by the wrapper.  No atomics.
    seg = tl.program_id(0)
    colj = tl.load(seg_col_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)
    hd_off = hh[:, None] * D + dd[None, :]
    k = tl.load(k_ptr + colj * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + colj * SH * D + hd_off, mask=hd_mask, other=0.0).to(tl.float32)
    dk_acc = tl.zeros([H_PAD, D_PAD], dtype=tl.float32)
    dv_acc = tl.zeros([H_PAD, D_PAD], dtype=tl.float32)

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        rows = tl.load(rowidx_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]
        goff = rows[:, None, None] * (SH * D) + hd_off[None, :, :]
        qb = tl.load(q_ptr + goff, mask=gmask, other=0.0).to(tl.float32)          # q_i
        dob = tl.load(do_ptr + goff, mask=gmask, other=0.0).to(tl.float32)        # dO_i
        smask = emask[:, None] & (hh[None, :] < H)
        soff = rows[:, None] * SH + hh[None, :]
        lb = tl.load(l_ptr + soff, mask=smask, other=1.0).to(tl.float32)           # l_i
        mb = tl.load(m_ptr + soff, mask=smask, other=0.0).to(tl.float32)           # m_i
        db = tl.load(dsum_ptr + soff, mask=smask, other=0.0).to(tl.float32)        # D_i
        prod = qb * k[None, :, :] * scale
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (SH * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                s = s + tl.load(bias_ptr + et[:, None] * SH + hh[None, :], mask=smask, other=0.0).to(tl.float32)
        dp = tl.sum(dob * v[None, :, :], axis=2) - db                               # <dO_i, v_j> - D_i
        if NORM == 1:
            inv = 1.0 / (lb + eps)
            p = tl.exp(tl.minimum(tl.maximum(s, lo), hi))
            p_hat = p * inv
            ds = p * dp * inv
            ds = tl.where((s >= lo) & (s <= hi), ds, 0.0)
        else:
            inv = 1.0 / tl.where(lb > 0, lb, 1.0)
            p_hat = tl.exp(s - mb) * inv
            ds = p_hat * dp
        ds = tl.where(emask[:, None], ds, 0.0)
        p_hat = tl.where(emask[:, None], p_hat, 0.0)
        dv_acc += tl.sum(p_hat[:, :, None] * dob, axis=0)
        if SCORE == 1:
            dk_acc += tl.sum((ds * scale)[:, :, None] * qb * gb, axis=0)
        else:
            dk_acc += tl.sum((ds * scale)[:, :, None] * qb, axis=0)

    tl.store(dk_ptr + seg * SH * D + hd_off, dk_acc, mask=hd_mask)
    tl.store(dv_ptr + seg * SH * D + hd_off, dv_acc, mask=hd_mask)


_SCORE = {"dot": 0, "gated": 1, "biased": 2}
_NORM = {"softmax": 0, "clampexp": 1}
_NSLOT = 256
# heads per program: the kernels take head-group SLICES of the [n, H, D] tensors (row stride SH = H); a program with
# 8 heads runs at BLOCK_E = 16 and costs 1.4x per edge-head and 2-3x per row / column in the backward compared with
# 4 heads (bench/kernel_constants.py), so wide models are processed in groups of HEADS_PER_PROGRAM heads.
HEADS_PER_PROGRAM = int(os.environ.get("GT3D_HEADS_PER_PROGRAM", "4"))


def _head_groups(h):
    g = max(1, HEADS_PER_PROGRAM)
    return [(g0, min(g, h - g0)) for g0 in range(0, h, g)]


def build_row_segments(crow: torch.Tensor, seg_edges: int):
    """Split every row's edge range into segments of at most seg_edges edges.

    Returns (seg_row, seg_start, seg_end, seg_ptr): seg_ptr is a CSR pointer from rows to
    their segments (segments are generated in row order), so per-row combines are
    torch_scatter.segment_csr reductions.  Rows with no edges get one empty segment so
    that every row owns at least one partial."""
    cache = getattr(crow, "_gt3d_seg_plans", None)
    if cache is None:
        cache = {}
        try:
            crow._gt3d_seg_plans = cache
        except AttributeError:
            pass
    plan = cache.get(seg_edges)
    if plan is not None:
        return plan
    deg = crow[1:] - crow[:-1]
    nseg = torch.clamp((deg + seg_edges - 1) // seg_edges, min=1)
    seg_ptr = torch.zeros(crow.numel(), dtype=torch.long, device=crow.device)
    seg_ptr[1:] = torch.cumsum(nseg, 0)
    n_rows = deg.numel()
    seg_row = torch.repeat_interleave(torch.arange(n_rows, device=crow.device), nseg)
    local = torch.arange(seg_row.numel(), device=crow.device) - seg_ptr[seg_row]
    seg_start = crow[seg_row] + local * seg_edges
    seg_end = torch.minimum(seg_start + seg_edges, crow[seg_row + 1])
    plan = (seg_row.to(torch.int32), seg_start, seg_end, seg_ptr)
    cache[seg_edges] = plan
    return plan



def build_csc(crow, col, etype, n_cols):
    """Transposed structure of the CSR block: (ccol [n_cols+1], rowidx [E], etype_csc [E]) with edges
    sorted by source column, cached per (crow, col) buffers."""
    cache = getattr(crow, "_gt3d_csc", None)
    if cache is None:
        cache = {}
        try:
            crow._gt3d_csc = cache
        except AttributeError:
            pass
    key = (col.data_ptr(), int(n_cols))
    csc = cache.get(key)
    if csc is not None:
        return csc
    n_rows = crow.numel() - 1
    rows = torch.repeat_interleave(torch.arange(n_rows, device=crow.device), crow[1:] - crow[:-1])
    colsl = col.long()
    perm = torch.argsort(colsl, stable=True)
    counts = torch.bincount(colsl, minlength=n_cols)
    ccol = torch.zeros(n_cols + 1, dtype=torch.long, device=crow.device)
    ccol[1:] = torch.cumsum(counts, 0)
    if col.numel() < 2 ** 31:
        ccol = ccol.to(torch.int32)                           # the kernels cast pointer loads to int64
    rowidx = rows[perm].to(torch.int32).contiguous()
    et = etype[perm].contiguous() if (etype is not None and etype.numel() == col.numel()) else None
    csc = (ccol, rowidx, et)
    cache[key] = csc
    return csc


def _tables(q, gate, bias, etype):
    dummy = torch.zeros(1, dtype=torch.float32, device=q.device)
    gate_c = gate.contiguous().float() if gate is not None else dummy
    bias_c = bias.contiguous().float() if bias is not None else dummy
    etype_c = etype.contiguous() if etype is not None else torch.zeros(1, dtype=torch.int32, device=q.device)
    return gate_c, bias_c, etype_c


def _attn_fwd_partial_rows(q, k, v, crow, col, etype=None, gate=None, bias=None, *, score="dot", norm="softmax",
                           clamp=(-5.0, 5.0), scale=None, seg_edges=512):
    """Unnormalised attention partials of the CSR block: acc[i] = sum_j p_ij v_j, l[i] = sum_j p_ij
    (with p = exp(s - m[i]) for softmax, m the row max; p = exp(clamp(s)) and m = 0 for clampexp).
    q: [N_rows, H, d]; k, v: [N_cols, H, d].  Returns fp32 (acc, l, m).  Partials of several blocks
    of the same rows combine like flash-decoding split-KV: m = max m_c; acc = sum acc_c e^{m_c-m}; ..."""
    from torch_scatter import segment_csr
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    seg_row, seg_start, seg_end, seg_ptr = build_row_segments(crow, seg_edges)
    n_seg = seg_row.numel()
    acc = torch.empty(n_seg, h, d, dtype=torch.float32, device=q.device)
    l_seg = torch.empty(n_seg, h, dtype=torch.float32, device=q.device)
    m_seg = torch.empty(n_seg, h, dtype=torch.float32, device=q.device)
    gate_c, bias_c, etype_c = _tables(q, gate, bias, etype)
    for g0, hg in _head_groups(h):
        if n_seg == 0:
            break
        sl = slice(g0, g0 + hg)
        _attn_fwd_kernel[(n_seg,)](
            q[:, sl], k[:, sl], v[:, sl], acc[:, sl], l_seg[:, sl], m_seg[:, sl], seg_row, seg_start, seg_end, col, etype_c,
            gate_c[:, sl] if gate is not None else gate_c, bias_c[:, sl] if bias is not None else bias_c,
            float(scale), float(clamp[0]), float(clamp[1]), 0.0, h,
            H=hg, D=d, H_PAD=_pow2(hg), D_PAD=_pow2(d),
            SCORE=_SCORE[score], NORM=_NORM[norm])
    if n_seg == n:
        l, m = l_seg, m_seg
    elif norm == "softmax":
        m = segment_csr(m_seg, seg_ptr, reduce="max")
        w = torch.exp(m_seg - m[seg_row.long()])
        l = segment_csr(l_seg * w, seg_ptr, reduce="sum")
        acc = segment_csr(acc * w.unsqueeze(-1), seg_ptr, reduce="sum")
    else:
        l = segment_csr(l_seg, seg_ptr, reduce="sum")
        acc = segment_csr(acc, seg_ptr, reduce="sum")
        m = m_seg[:0]
    if norm == "softmax":
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))       # empty rows
    else:
        m = torch.zeros_like(l)
    return acc, l, m


def normalize(acc, l, norm, eps=1e-6, dtype=None):
    if norm == "softmax":
        o = acc / torch.where(l > 0, l, torch.ones_like(l)).unsqueeze(-1)
    else:
        o = acc / (l + eps).unsqueeze(-1)
    return o.to(dtype) if dtype is not None else o


def attn_bwd(q, k, v, crow, col, etype, gate, bias, o, l, m, do, *, score="dot", norm="softmax",
             clamp=(-5.0, 5.0), eps=1e-6, scale=None, seg_edges=512, atomic_kv=False):
    """Backward of the normalised attention given the (possibly global) per-row statistics
    l, m and the normalised output o of the rows.  Returns (dq, dk, dv, dgate, dbias) in fp32.
    dK/dV come from a deterministic column (CSC) pass unless atomic_kv=True."""
    from torch_scatter import segment_csr
    n, h, d = q.shape
    n_cols = k.shape[0]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v, o, do = q.contiguous(), k.contiguous(), v.contiguous(), o.contiguous(), do.contiguous()
    l, m = l.contiguous().float(), m.contiguous().float()
    assert not atomic_kv, "the atomic dK/dV path was removed with the head-group kernels"
    dsum = (do.float() * o.float()).sum(-1).contiguous()                           # D_i  [N_rows, H]
    dq, dgate, dbias = _bwd_row_pass_rows(q, k, v, crow, col, etype, gate, bias, o, do, l, m, score=score, norm=norm,
                                          clamp=clamp, eps=eps, scale=scale, partial=False, seg_edges=seg_edges)
    dk, dv = _bwd_col_pass_rows(q, k, v, crow, col, etype, gate, bias, do, l, m, dsum, score=score, norm=norm,
                                clamp=clamp, eps=eps, scale=scale, seg_edges=seg_edges)
    return dq, dk, dv, dgate, dbias


def _bwd_row_pass_rows(q, k, v, crow, col, etype, gate, bias, o_or_negdl, do, l, m, *, score, norm, clamp, eps, scale,
                       partial, seg_edges):
    """Row pass of the per-row backward kernels over the CSR: dq [n, h, d] (+ dgate, dbias), head group by head group.
    partial=True: o_or_negdl = -dl [n, h] and do = dacc (gradients of the unnormalised partials), l = ones."""
    from torch_scatter import segment_csr
    n, h, d = q.shape
    seg_row, seg_start, seg_end, seg_ptr = build_row_segments(crow, seg_edges)
    n_seg = seg_row.numel()
    dq_seg = torch.empty(n_seg, h, d, dtype=torch.float32, device=q.device)
    gate_c, bias_c, etype_c = _tables(q, gate, bias, etype)
    t = gate_c.shape[0] if (gate is not None and score == "gated") else (bias_c.shape[0] if (bias is not None and score == "biased") else 1)
    dummy = torch.zeros(1, dtype=torch.float32, device=q.device)
    dgates, dbiases = [], []
    for g0, hg in _head_groups(h):
        sl = slice(g0, g0 + hg)
        dgate_slots = torch.zeros(_NSLOT, t, hg, d, dtype=torch.float32, device=q.device)
        dbias_slots = torch.zeros(_NSLOT, t, hg, dtype=torch.float32, device=q.device)
        if n_seg > 0:
            o_arg = o_or_negdl[:, sl] if partial else o_or_negdl[:, sl]
            _attn_bwd_kernel[(n_seg,)](
                q[:, sl], k[:, sl], v[:, sl], o_arg, do[:, sl], l[:, sl], m[:, sl], dq_seg[:, sl], dummy, dummy,
                dgate_slots, dbias_slots, seg_row, seg_start, seg_end, col, etype_c,
                gate_c[:, sl] if gate is not None else gate_c, bias_c[:, sl] if bias is not None else bias_c,
                float(scale), float(clamp[0]), float(clamp[1]), float(eps), h,
                NSLOT=_NSLOT, T=t, H=hg, D=d, H_PAD=_pow2(hg), D_PAD=_pow2(d),
                SCORE=_SCORE[score], NORM=_NORM[norm], WITH_KV=False, PARTIAL=partial)
        dgates.append(dgate_slots.sum(0)); dbiases.append(dbias_slots.sum(0))
    dq = dq_seg if n_seg == n else segment_csr(dq_seg, seg_ptr, reduce="sum")
    dgate = torch.cat(dgates, dim=1) if (gate is not None and score == "gated") else None
    dbias = torch.cat(dbiases, dim=1) if (bias is not None and score == "biased") else None
    return dq, dgate, dbias


def _bwd_col_pass_rows(q, k, v, crow, col, etype, gate, bias, do, l, m, dsum, *, score, norm, clamp, eps, scale, seg_edges):
    """Column pass of the per-row backward kernels over the CSC: dk, dv [n_cols, h, d], head group by head group
    (do = dO or dacc, l = row sums or ones, dsum = D_i or -dl_i)."""
    from torch_scatter import segment_csr
    n, h, d = q.shape
    n_cols = k.shape[0]
    ccol, rowidx, et_csc = build_csc(crow, col, etype, n_cols)
    cseg_col, cseg_start, cseg_end, cseg_ptr = build_row_segments(ccol, seg_edges)
    n_cseg = cseg_col.numel()
    dk_seg = torch.empty(n_cseg, h, d, dtype=torch.float32, device=q.device)
    dv_seg = torch.empty(n_cseg, h, d, dtype=torch.float32, device=q.device)
    gate_c, bias_c, _ = _tables(q, gate, bias, etype)
    et_arg = et_csc if et_csc is not None else torch.zeros(1, dtype=torch.int32, device=q.device)
    for g0, hg in _head_groups(h):
        if n_cseg == 0:
            break
        sl = slice(g0, g0 + hg)
        _attn_bwd_kv_kernel[(n_cseg,)](
            q[:, sl], k[:, sl], v[:, sl], do[:, sl], l[:, sl], m[:, sl], dsum[:, sl], dk_seg[:, sl], dv_seg[:, sl],
            cseg_col, cseg_start, cseg_end, rowidx, et_arg,
            gate_c[:, sl] if gate is not None else gate_c, bias_c[:, sl] if bias is not None else bias_c,
            float(scale), float(clamp[0]), float(clamp[1]), float(eps), h,
            H=hg, D=d, H_PAD=_pow2(hg), D_PAD=_pow2(d),
            SCORE=_SCORE[score], NORM=_NORM[norm])
    if n_cseg == n_cols:
        return dk_seg, dv_seg
    return segment_csr(dk_seg, cseg_ptr, reduce="sum"), segment_csr(dv_seg, cseg_ptr, reduce="sum")


def _attn_bwd_partial_rows(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, *, score="dot", norm="softmax",
                           clamp=(-5.0, 5.0), scale=None, seg_edges=512):
    """Backward of attn_fwd_partial: gradients w.r.t. q, k, v, gate, bias given dacc [N_rows,H,d] and
    dl [N_rows,H] (gradients of the unnormalised partials acc, l) and the partial's row max m."""
    args = _partial_args(q, k, v, m, dacc, dl, scale)
    dq, dgate, dbias = _bwd_row_pass_rows(*args[:3], crow, col, etype, gate, bias, args[5], args[4], args[6], args[3],
                                          score=score, norm=norm, clamp=clamp, eps=0.0, scale=args[7], partial=True, seg_edges=seg_edges)
    dk, dv = _bwd_col_pass_rows(*args[:3], crow, col, etype, gate, bias, args[4], args[6], args[3], args[5],
                                score=score, norm=norm, clamp=clamp, eps=0.0, scale=args[7], seg_edges=seg_edges)
    return dq, dk, dv, dgate, dbias


def _partial_args(q, k, v, m, dacc, dl, scale):
    """(q, k, v, m, dacc, -dl, ones, scale) for the backward of the unnormalised partials.  dacc keeps its dtype
    (bf16 halves the per-edge reads of the column pass; the kernels upcast)."""
    d = q.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    m = m.contiguous().float()
    return (q.contiguous(), k.contiguous(), v.contiguous(), m, dacc.contiguous(), (-dl).contiguous().float(),
            torch.ones_like(m), scale)


# ---- dispatch: per-row kernels for dense blocks, row-tiled kernels (triton_tiled) for blocks with few edges per row ----
# The per-row kernels cost ~3.5 ns/row (fwd) / ~10 ns/row (bwd) regardless of the edge count; the tiled kernels are
# 1.7-2.3x faster below ~2 edges per row (streamed column sub-blocks) and slower above ~3 (one-hot tl.dot per edge).
TILED_THRESHOLD = float(os.environ.get("GT3D_TILED_THRESHOLD", "3.5"))


def _use_tiled(crow, col):
    n_rows = crow.numel() - 1
    return n_rows > 0 and col.numel() / n_rows < TILED_THRESHOLD


def _use_tiled_cols(col, n_cols):
    """The backward column pass is dispatched on the edges per COLUMN: a block with 13 edges per row can have 3 per
    column (64 GPUs 16x4: 153K rows x 612K columns), where the per-column cost of the per-row kernels dominates."""
    return n_cols > 0 and col.numel() / n_cols < TILED_THRESHOLD


def attn_fwd_partial(q, k, v, crow, col, etype=None, gate=None, bias=None, *, score="dot", norm="softmax",
                     clamp=(-5.0, 5.0), scale=None, seg_edges=512):
    if _use_tiled(crow, col):
        from gt3d.attention import triton_tiled as tt
        return tt.attn_fwd_partial(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm, clamp=clamp, scale=scale)
    return _attn_fwd_partial_rows(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm, clamp=clamp, scale=scale,
                                  seg_edges=seg_edges)


def attn_bwd_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, *, score="dot", norm="softmax",
                     clamp=(-5.0, 5.0), scale=None, seg_edges=512, dq_accum=None):
    """(dq, dk, dv, dgate, dbias) of the unnormalised partials.  The row pass (CSR) and the column pass (CSC) are
    dispatched separately between the per-row and the tiled kernels (edges per row / per column vs TILED_THRESHOLD).
    dq_accum [n, h, d] (or [n, h*d]): dq is added in place (compact-accumulate row pass when tiled) and the returned
    dq is None."""
    from gt3d.attention import triton_tiled as tt
    args = _partial_args(q, k, v, m, dacc, dl, scale)
    q_, k_, v_, m_, dacc_, neg_dl, ones, scale = args
    n_cols = k_.shape[0]
    if _use_tiled(crow, col):
        dq, dgate, dbias = tt.bwd_row_partial(q_, k_, v_, crow, col, etype, gate, bias, m_, dacc_, neg_dl, score=score, norm=norm,
                                              clamp=clamp, scale=scale, dq_accum=dq_accum)
    else:
        dq, dgate, dbias = _bwd_row_pass_rows(q_, k_, v_, crow, col, etype, gate, bias, neg_dl, dacc_, ones, m_, score=score,
                                              norm=norm, clamp=clamp, eps=0.0, scale=scale, partial=True, seg_edges=seg_edges)
        if dq_accum is not None:
            dq_accum.view(dq_accum.shape[0], -1).add_(dq.view(dq.shape[0], -1).to(dq_accum.dtype))
            dq = None
    if _use_tiled_cols(col, n_cols):
        dk, dv = tt.bwd_col_partial(q_, k_, v_, crow, col, etype, gate, bias, m_, dacc_, neg_dl, score=score, norm=norm,
                                    clamp=clamp, scale=scale)
    else:
        dk, dv = _bwd_col_pass_rows(q_, k_, v_, crow, col, etype, gate, bias, dacc_, ones, m_, neg_dl, score=score, norm=norm,
                                    clamp=clamp, eps=0.0, scale=scale, seg_edges=seg_edges)
    return dq, dk, dv, dgate, dbias


def attn_fwd_partial_accum(q, k, v, crow, col, etype, gate, bias, acc_buf, l_buf, *, score="dot", norm="clampexp",
                           clamp=(-5.0, 5.0), scale=None, seg_edges=512):
    """acc_buf [n, h*d] += acc, l_buf [n, h] += l of the block's unnormalised clamp-exp partials (row-strided views
    allowed).  Sparse blocks (streamed sub-blocks): the compact-accumulate tiled kernels, which tile only the rows
    that own an edge and add in place; dense blocks: the per-row kernels and a plain add."""
    if _use_tiled(crow, col):
        from gt3d.attention import triton_tiled as tt
        tt.attn_fwd_partial(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm, clamp=clamp, scale=scale,
                            accum=(acc_buf, l_buf))
        return
    acc, l, _ = _attn_fwd_partial_rows(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm, clamp=clamp,
                                       scale=scale, seg_edges=seg_edges)
    acc_buf += acc.view(acc.shape[0], -1).to(acc_buf.dtype)
    l_buf += l.to(l_buf.dtype)


def attn_bwd_partial_accum(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, dq_buf, *, score="dot", norm="clampexp",
                           clamp=(-5.0, 5.0), scale=None, seg_edges=512):
    """dq_buf [n, h, d] += dq of the block; returns (dk, dv, dgate, dbias) (see attn_bwd_partial)."""
    _, dk, dv, dgate, dbias = attn_bwd_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, score=score, norm=norm,
                                               clamp=clamp, scale=scale, seg_edges=seg_edges, dq_accum=dq_buf)
    return dk, dv, dgate, dbias


class _MaskedSparseAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, crow, col, etype, gate, bias, score, norm, lo, hi, eps, scale, seg_edges):
        acc, l, m = attn_fwd_partial(q, k, v, crow, col, etype, gate, bias, score=score, norm=norm,
                                     clamp=(lo, hi), scale=scale, seg_edges=seg_edges)
        o = normalize(acc, l, norm, eps, dtype=q.dtype)
        ctx.save_for_backward(q, k, v, o, l, m,
                              gate if gate is not None else torch.zeros(0, device=q.device),
                              bias if bias is not None else torch.zeros(0, device=q.device))
        ctx.graph = (crow, col, etype)            # same Python objects -> cached segment plans / CSC survive
        ctx.cfg = (score, norm, lo, hi, eps, scale, seg_edges, gate is not None, bias is not None)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m, gate, bias = ctx.saved_tensors
        crow, col, etype = ctx.graph
        score, norm, lo, hi, eps, scale, seg_edges, has_gate, has_bias = ctx.cfg
        dq, dk, dv, dgate, dbias = attn_bwd(q, k, v, crow, col, etype,
                                            gate if has_gate else None, bias if has_bias else None,
                                            o, l, m, do, score=score, norm=norm, clamp=(lo, hi), eps=eps,
                                            scale=scale, seg_edges=seg_edges)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None,
                dgate.to(gate.dtype) if dgate is not None else None,
                dbias.to(bias.dtype) if dbias is not None else None,
                None, None, None, None, None, None, None)


def masked_sparse_attention_triton(q, k, v, crow, col, etype=None, *, gate=None, bias=None,
                                   score="dot", norm="softmax", clamp=(-5.0, 5.0), eps=1e-6, scale=None,
                                   seg_edges=512):
    """Backend B (Triton prototype) entry point; same signature as backend_torch plus
    seg_edges, the maximum number of edges one program handles (rows longer than this are
    split across programs and combined afterwards)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if score in ("gated", "biased"):
        assert etype is not None
        etype = etype.contiguous()
    return _MaskedSparseAttentionTriton.apply(q.contiguous(), k.contiguous(), v.contiguous(),
                                             crow.contiguous(), col.contiguous(), etype, gate, bias,
                                             score, norm, float(clamp[0]), float(clamp[1]), float(eps), float(scale),
                                             int(seg_edges))
