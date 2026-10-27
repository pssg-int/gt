"""Row-tiled Triton kernels for masked sparse attention.

The kernels in triton_kernel.py run one program per (row, edge segment).  A program's cost is dominated by its
dependent-load latency chain, ~3.5 ns/row forward and ~10 ns/row backward, independent of the row's edge count;
that is fine for dense blocks (10+ edges per row) but dominates when a block is sparse per row, which is exactly the
regime of the streamed layer (a column sub-block leaves 1-3 edges per row) and of the backward column pass.

Here one program handles a TILE of BLOCK_R consecutive rows: it streams the tile's edges (a contiguous CSR range) in
BLOCK_E chunks, finds each edge's local row by comparing the edge index with the tile's row pointers, gathers the
edge's q row from the register tile with a one-hot tl.dot (exact, IEEE fp32) and reduces p_ij v_j back onto the rows
with the transposed one-hot dot.  Long tiles are split into segments of at most seg_edges edges; the first segment of
a tile writes to the row slots, later segments to extra slots that the wrapper adds back (flash-decoding rescale for
softmax).  The backward column pass tiles the CSC the same way (columns of a sub-block are just as sparse).

Compact-accumulate mode (CA, `accum=` / `dq_accum=`): a streamed column sub-block is hypersparse (papers100M at
64 GPUs: 0.3-0.6 edges per row of the row block), so the tiles are built on the doubly compressed CSR (build_dcsr:
only the rows that own an edge in the sub-block), every program loads / stores its rows through the row list, and
the partials are added in place to the caller's accumulators (the streamed layer's packed buffer / dq) instead of
being written to a fresh full-row buffer and added afterwards.
"""
import math
import os

import torch
import triton
import triton.language as tl

_SCORE = {"dot": 0, "gated": 1, "biased": 2}
_NORM = {"softmax": 0, "clampexp": 1}
_NSLOT = 256
# precision of the one-hot gathers / segmented reductions (tl.dot with a 0/1 operand), GT3D_TILED_PREC:
#   ieee   : fp32 on the FMA units (exact; ~8 KFLOP per edge, the tiled kernels' cost on dense rows)
#   tf32   : one tensor-core dot, data operand rounded to 10 mantissa bits (~5e-4 relative; below bf16 communication)
#   tf32x2 : two tensor-core dots on the hi / lo split of the data operand (~2^-21 relative, fp32-like)
_PREC = {"ieee": 0, "tf32": 1, "tf32x2": 2}
TILED_PREC = _PREC[os.environ.get("GT3D_TILED_PREC", "ieee")]


def set_precision(name):
    global TILED_PREC
    TILED_PREC = _PREC[name]


@triton.jit
def _dot1h(a, b, PREC: tl.constexpr):
    """a @ b for an exact 0/1 matrix a (one-hot gather or segmented reduction); see TILED_PREC."""
    if PREC == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif PREC == 1:
        return tl.dot(a, b, input_precision="tf32")
    else:
        b_hi = (b.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)     # low 13 mantissa bits cleared
        return tl.dot(a, b_hi, input_precision="tf32") + tl.dot(a, b - b_hi, input_precision="tf32")


def _pow2(x):
    return 1 << (int(x) - 1).bit_length()


def _configs():
    return [triton.Config({"BLOCK_E": be}, num_warps=nw) for be in (16, 32, 64) for nw in (4, 8)]


# ------------------------------------------------------------------------------------------------------------
# tile plan
# ------------------------------------------------------------------------------------------------------------
def build_tiles(crow: torch.Tensor, block_r: int, seg_edges: int):
    """Tiles of block_r consecutive rows; a tile whose edge range exceeds seg_edges is cut into several segments.
    Returns dict(seg_row0 [S] i32, seg_start [S] i64, seg_end [S] i64, seg_out [S] i64 (first output slot),
    n_tiles, n_main = n_tiles*block_r, extra_rows [n_extra*block_r] i64 (row of every extra slot, or -1)) cached
    on the crow tensor."""
    cache = getattr(crow, "_gt3d_tiles", None)
    if cache is None:
        cache = {}
        try:
            crow._gt3d_tiles = cache
        except AttributeError:
            pass
    key = (block_r, seg_edges)
    plan = cache.get(key)
    if plan is not None:
        return plan
    dev = crow.device
    n = crow.numel() - 1
    n_tiles = -(-n // block_r)
    bounds = torch.full((n_tiles + 1,), int(crow[-1]), dtype=torch.long, device=dev)
    idx = torch.arange(n_tiles + 1, device=dev) * block_r
    bounds[idx <= n] = crow[idx[idx <= n]].long()
    t_start, t_end = bounds[:-1], bounds[1:]
    nseg = torch.clamp((t_end - t_start + seg_edges - 1) // seg_edges, min=1)
    seg_ptr = torch.zeros(n_tiles + 1, dtype=torch.long, device=dev)
    seg_ptr[1:] = torch.cumsum(nseg, 0)
    n_seg = int(seg_ptr[-1])
    seg_tile = torch.repeat_interleave(torch.arange(n_tiles, device=dev), nseg)
    local = torch.arange(n_seg, device=dev) - seg_ptr[seg_tile]
    seg_start = t_start[seg_tile] + local * seg_edges
    seg_end = torch.minimum(seg_start + seg_edges, t_end[seg_tile])
    is_extra = local > 0
    n_main = n_tiles * block_r
    extra_idx = torch.cumsum(is_extra.long(), 0) - 1                   # index among extra segments
    seg_out = torch.where(is_extra, n_main + extra_idx * block_r, seg_tile * block_r)
    extra_tiles = seg_tile[is_extra]
    extra_rows = (extra_tiles[:, None] * block_r + torch.arange(block_r, device=dev)[None, :]).reshape(-1)
    extra_rows = torch.where(extra_rows < n, extra_rows, torch.full_like(extra_rows, -1))
    plan = dict(seg_row0=(seg_tile * block_r).to(torch.int32), seg_start=seg_start, seg_end=seg_end, seg_out=seg_out,
                n_tiles=n_tiles, n_main=n_main, n_extra=int(is_extra.sum()), extra_rows=extra_rows, n_seg=n_seg)
    cache[key] = plan
    return plan


def build_dcsr(crow: torch.Tensor):
    """Doubly compressed CSR (hypersparse rows): rows_c = the rows that own at least one edge (int32, ascending),
    crow_c = their row pointers (int32 [len(rows_c) + 1]; the edges of consecutive non-empty rows stay contiguous
    because the skipped rows own none).  Cached on the crow tensor."""
    cache = getattr(crow, "_gt3d_dcsr", None)
    if cache is not None:
        return cache
    deg = crow[1:] - crow[:-1]
    rows_c = torch.nonzero(deg > 0).squeeze(1)
    crow_c = torch.cat([crow[rows_c], crow[-1:]]).contiguous()
    cache = (rows_c.to(torch.int32).contiguous(), crow_c)
    try:
        crow._gt3d_dcsr = cache
    except AttributeError:
        pass
    return cache


def _combine(main, extra, extra_rows, n, norm, m_main=None, m_extra=None):
    """Add the extra slots back onto their rows (clamp-exp: plain sums; softmax: rescale to the row max).
    main/extra: [slots, h, d] or [slots, h]; m_*: [slots, h] per-head row maxima."""
    if extra.shape[0] == 0:
        return main[:n], (m_main[:n] if m_main is not None else None)
    valid = extra_rows >= 0
    rows = extra_rows[valid]
    ex = extra[valid]
    if norm == "clampexp":
        out = main.clone()
        out.index_add_(0, rows, ex)
        return out[:n], None
    m_glob = m_main.clone()
    m_glob.index_reduce_(0, rows, m_extra[valid], "amax", include_self=True)
    w_main = torch.exp(m_main - m_glob)
    w_main = torch.where(torch.isfinite(w_main), w_main, torch.zeros_like(w_main))
    w_ex = torch.exp(m_extra[valid] - m_glob[rows])
    w_ex = torch.where(torch.isfinite(w_ex), w_ex, torch.zeros_like(w_ex))
    if main.dim() == 3:
        w_main, w_ex = w_main.unsqueeze(-1), w_ex.unsqueeze(-1)
    out = (main * w_main).clone()
    out.index_add_(0, rows, ex * w_ex)
    return out[:n], m_glob[:n]


# ------------------------------------------------------------------------------------------------------------
# forward: unnormalised partials of a tile of rows
# ------------------------------------------------------------------------------------------------------------
@triton.autotune(configs=_configs(), key=["H", "D", "SCORE", "NORM", "BLOCK_R", "PREC", "CA"],
                 restore_value=["acc_ptr", "l_ptr"])                    # CA accumulates in place: undo the tuning runs
@triton.jit
def _fwd_tiled_kernel(
    q_ptr, k_ptr, v_ptr, acc_ptr, l_ptr, m_ptr, xacc_ptr, xl_ptr, crow_ptr, rows_ptr,
    seg_row0_ptr, seg_start_ptr, seg_end_ptr, seg_out_ptr, col_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, n_rows, n_main, acc_stride, l_stride,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr, HD_PAD: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, PREC: tl.constexpr,
    CA: tl.constexpr,
):
    # CA (compact-accumulate): crow_ptr / n_rows / the tile plan are those of the compact CSR, rows_ptr maps a
    # compact row to its global row (q rows, accumulator rows); main segments add their tile in place to acc_ptr /
    # l_ptr (row strides acc_stride / l_stride, clamp-exp only: no running max), extra segments (out0 >= n_main)
    # write to the extra slots xacc_ptr / xl_ptr which the wrapper adds.  Not CA: the original single-buffer layout.
    seg = tl.program_id(0)
    row0 = tl.load(seg_row0_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)
    out0 = tl.load(seg_out_ptr + seg).to(tl.int64)

    rr = tl.arange(0, BLOCK_R)
    ci = row0 + rr                                                         # row index in the tile plan's space
    rmask = ci < n_rows
    rowend = tl.load(crow_ptr + ci + 1, mask=rmask, other=0).to(tl.int64)
    rowend = tl.where(rmask, rowend, end)                                  # rows past the end own no edge
    if CA:
        rows = tl.load(rows_ptr + ci, mask=rmask, other=0).to(tl.int64)    # global row of every compact row
    else:
        rows = ci

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)                        # [H_PAD, D_PAD]
    hd_off = hh[:, None] * D + dd[None, :]
    q = tl.load(q_ptr + rows[:, None, None] * (H * D) + hd_off[None, :, :],
                mask=rmask[:, None, None] & hd_mask[None, :, :], other=0.0).to(tl.float32)   # [BR, H_PAD, D_PAD]
    q2 = tl.reshape(q, [BLOCK_R, HD_PAD])

    acc = tl.zeros([BLOCK_R, HD_PAD], dtype=tl.float32)
    l = tl.zeros([BLOCK_R, H_PAD], dtype=tl.float32)
    m = tl.full([BLOCK_R, H_PAD], float("-inf"), dtype=tl.float32)

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        erow = tl.sum((offs[:, None] >= rowend[None, :]).to(tl.int32), axis=1)          # local row of every edge
        onehot_t = ((erow[:, None] == rr[None, :]) & emask[:, None]).to(tl.float32)   # [BE, BR]
        onehot = tl.trans(onehot_t)                                                    # [BR, BE]
        qe = tl.reshape(_dot1h(onehot_t, q2, PREC), [BLOCK_E, H_PAD, D_PAD])
        cols = tl.load(col_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]
        goff = cols[:, None, None] * (H * D) + hd_off[None, :, :]
        kb = tl.load(k_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        prod = qe * kb * scale
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (H * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)                                               # [BE, H_PAD]
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                bmask = emask[:, None] & (hh[None, :] < H)
                s = s + tl.load(bias_ptr + et[:, None] * H + hh[None, :], mask=bmask, other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        if NORM == 1:
            p = tl.exp(tl.minimum(tl.maximum(s, lo), hi))
            p = tl.where(emask[:, None], p, 0.0)
            pv = tl.reshape(p[:, :, None] * vb, [BLOCK_E, HD_PAD])
            acc += _dot1h(onehot, pv, PREC)
            l += tl.sum(onehot[:, :, None] * p[None, :, :], axis=1)
        else:
            s = tl.where(emask[:, None], s, float("-inf"))
            s_r = tl.where(onehot[:, :, None] > 0, s[None, :, :], float("-inf"))       # [BR, BE, H_PAD]
            m_new = tl.maximum(m, tl.max(s_r, axis=1))
            alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
            m_e = tl.sum(tl.where(onehot_t[:, :, None] > 0, m_new[None, :, :], 0.0), axis=1)   # [BE, H_PAD] row max per edge (0 * -inf = nan otherwise)
            p = tl.exp(s - m_e)
            p = tl.where(emask[:, None], p, 0.0)
            pv = tl.reshape(p[:, :, None] * vb, [BLOCK_E, HD_PAD])
            l = l * alpha + tl.sum(onehot[:, :, None] * p[None, :, :], axis=1)
            alpha3 = tl.reshape(tl.broadcast_to(alpha[:, :, None], [BLOCK_R, H_PAD, D_PAD]), [BLOCK_R, HD_PAD])
            acc = acc * alpha3 + _dot1h(onehot, pv, PREC)
            m = m_new

    smask = rmask[:, None, None] & hd_mask[None, :, :]
    hmask = rmask[:, None] & (hh[None, :] < H)
    acc3 = tl.reshape(acc, [BLOCK_R, H_PAD, D_PAD])
    if CA:
        segm = tl.zeros([BLOCK_R], dtype=tl.int64) + out0                 # the segment's out0 as a block
        is_main = segm < n_main
        is_extra = segm >= n_main
        mmask = smask & is_main[:, None, None]
        pa = acc_ptr + rows[:, None, None] * acc_stride + hd_off[None, :, :]
        tl.store(pa, (tl.load(pa, mask=mmask, other=0.0).to(tl.float32) + acc3).to(acc_ptr.dtype.element_ty), mask=mmask)
        lmask = hmask & is_main[:, None]
        pl = l_ptr + rows[:, None] * l_stride + hh[None, :]
        tl.store(pl, (tl.load(pl, mask=lmask, other=0.0).to(tl.float32) + l).to(l_ptr.dtype.element_ty), mask=lmask)
        xs = out0 - n_main + rr
        tl.store(xacc_ptr + xs[:, None, None] * (H * D) + hd_off[None, :, :], acc3, mask=smask & is_extra[:, None, None])
        tl.store(xl_ptr + xs[:, None] * H + hh[None, :], l, mask=hmask & is_extra[:, None])
    else:
        slots = out0 + rr
        tl.store(acc_ptr + slots[:, None, None] * (H * D) + hd_off[None, :, :], acc3, mask=smask)
        tl.store(l_ptr + slots[:, None] * H + hh[None, :], l, mask=hmask)
        tl.store(m_ptr + slots[:, None] * H + hh[None, :], m, mask=hmask)


# ------------------------------------------------------------------------------------------------------------
# backward row pass: dQ for a tile of rows (+ gate / bias gradients into NSLOT accumulation slots)
# ------------------------------------------------------------------------------------------------------------
@triton.autotune(configs=_configs(), key=["H", "D", "SCORE", "NORM", "BLOCK_R", "PARTIAL", "PREC", "CA"],
                 reset_to_zero=["dgate_ptr", "dbias_ptr"], restore_value=["dq_ptr"])
@triton.jit
def _bwd_row_tiled_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, m_ptr, dq_ptr, xdq_ptr, dgate_ptr, dbias_ptr, crow_ptr, rows_ptr,
    seg_row0_ptr, seg_start_ptr, seg_end_ptr, seg_out_ptr, col_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, eps, n_rows, n_main, dq_stride,
    NSLOT: tl.constexpr, T: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr, HD_PAD: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, PARTIAL: tl.constexpr,
    PREC: tl.constexpr, CA: tl.constexpr,
):
    # PARTIAL: upstream gradient w.r.t. the unnormalised partials: do_ptr = dacc, o_ptr = -dl (per row, [n, H]), l = 1.
    # CA: compact rows (rows_ptr -> global row for the q / o / do / l / m loads), dq added in place to dq_ptr (row
    # stride dq_stride) by the main segments, extra segments to xdq_ptr (see _fwd_tiled_kernel).
    seg = tl.program_id(0)
    row0 = tl.load(seg_row0_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)
    out0 = tl.load(seg_out_ptr + seg).to(tl.int64)

    rr = tl.arange(0, BLOCK_R)
    ci = row0 + rr
    rmask = ci < n_rows
    rowend = tl.load(crow_ptr + ci + 1, mask=rmask, other=0).to(tl.int64)
    rowend = tl.where(rmask, rowend, end)
    if CA:
        rows = tl.load(rows_ptr + ci, mask=rmask, other=0).to(tl.int64)
    else:
        rows = ci

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hmask2 = rmask[:, None] & (hh[None, :] < H)
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)
    hd_off = hh[:, None] * D + dd[None, :]
    tmask = rmask[:, None, None] & hd_mask[None, :, :]
    roff = rows[:, None, None] * (H * D) + hd_off[None, :, :]
    q2 = tl.reshape(tl.load(q_ptr + roff, mask=tmask, other=0.0).to(tl.float32), [BLOCK_R, HD_PAD])
    do2 = tl.reshape(tl.load(do_ptr + roff, mask=tmask, other=0.0).to(tl.float32), [BLOCK_R, HD_PAD])
    if PARTIAL:
        dsum = tl.load(o_ptr + rows[:, None] * H + hh[None, :], mask=hmask2, other=0.0)       # -dl  [BR, H_PAD]
        inv = tl.zeros([BLOCK_R, H_PAD], dtype=tl.float32) + 1.0
        m_r = tl.load(m_ptr + rows[:, None] * H + hh[None, :], mask=hmask2, other=0.0)
    else:
        o = tl.load(o_ptr + roff, mask=tmask, other=0.0).to(tl.float32)
        do3 = tl.load(do_ptr + roff, mask=tmask, other=0.0).to(tl.float32)
        dsum = tl.sum(do3 * o, axis=2)                                                          # D_i [BR, H_PAD]
        l_r = tl.load(l_ptr + rows[:, None] * H + hh[None, :], mask=hmask2, other=1.0)
        m_r = tl.load(m_ptr + rows[:, None] * H + hh[None, :], mask=hmask2, other=0.0)
        if NORM == 1:
            inv = 1.0 / (l_r + eps)
        else:
            inv = 1.0 / tl.where(l_r > 0, l_r, 1.0)

    dq_acc = tl.zeros([BLOCK_R, HD_PAD], dtype=tl.float32)
    slot = seg % NSLOT

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        erow = tl.sum((offs[:, None] >= rowend[None, :]).to(tl.int32), axis=1)
        onehot_t = ((erow[:, None] == rr[None, :]) & emask[:, None]).to(tl.float32)   # [BE, BR]
        onehot = tl.trans(onehot_t)
        qe = tl.reshape(_dot1h(onehot_t, q2, PREC), [BLOCK_E, H_PAD, D_PAD])
        doe = tl.reshape(_dot1h(onehot_t, do2, PREC), [BLOCK_E, H_PAD, D_PAD])
        dsum_e = tl.sum(onehot_t[:, :, None] * dsum[None, :, :], axis=1)                # [BE, H_PAD]
        inv_e = tl.sum(onehot_t[:, :, None] * inv[None, :, :], axis=1)
        m_e = tl.sum(onehot_t[:, :, None] * m_r[None, :, :], axis=1)
        cols = tl.load(col_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]
        goff = cols[:, None, None] * (H * D) + hd_off[None, :, :]
        kb = tl.load(k_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        prod = qe * kb * scale
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (H * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                bmask = emask[:, None] & (hh[None, :] < H)
                s = s + tl.load(bias_ptr + et[:, None] * H + hh[None, :], mask=bmask, other=0.0).to(tl.float32)
        dp = tl.sum(doe * vb, axis=2) - dsum_e
        if NORM == 1:
            p = tl.exp(tl.minimum(tl.maximum(s, lo), hi))
            ds = p * dp * inv_e
            ds = tl.where((s >= lo) & (s <= hi), ds, 0.0)
        else:
            p_hat = tl.exp(s - m_e) * inv_e
            ds = p_hat * dp
        ds = tl.where(emask[:, None], ds, 0.0)
        ds_s = ds * scale
        if SCORE == 1:
            dq_acc += _dot1h(onehot, tl.reshape(ds_s[:, :, None] * kb * gb, [BLOCK_E, HD_PAD]), PREC)
            dg = ds[:, :, None] * prod
            for t in tl.static_range(T):
                sel = tl.where((et == t)[:, None, None], dg, 0.0)
                tl.atomic_add(dgate_ptr + (slot * T + t) * (H * D) + hd_off, tl.sum(sel, axis=0), mask=hd_mask)
        else:
            dq_acc += _dot1h(onehot, tl.reshape(ds_s[:, :, None] * kb, [BLOCK_E, HD_PAD]), PREC)
            if SCORE == 2:
                for t in tl.static_range(T):
                    sel = tl.where((et == t)[:, None], ds, 0.0)
                    tl.atomic_add(dbias_ptr + (slot * T + t) * H + hh, tl.sum(sel, axis=0), mask=hh < H)

    dq3 = tl.reshape(dq_acc, [BLOCK_R, H_PAD, D_PAD])
    if CA:
        segm = tl.zeros([BLOCK_R], dtype=tl.int64) + out0
        is_main = segm < n_main
        mmask = tmask & is_main[:, None, None]
        pd = dq_ptr + rows[:, None, None] * dq_stride + hd_off[None, :, :]
        tl.store(pd, (tl.load(pd, mask=mmask, other=0.0).to(tl.float32) + dq3).to(dq_ptr.dtype.element_ty), mask=mmask)
        xs = out0 - n_main + rr
        tl.store(xdq_ptr + xs[:, None, None] * (H * D) + hd_off[None, :, :], dq3, mask=tmask & (segm >= n_main)[:, None, None])
    else:
        slots = out0 + rr
        tl.store(dq_ptr + slots[:, None, None] * (H * D) + hd_off[None, :, :], dq3, mask=tmask)


# ------------------------------------------------------------------------------------------------------------
# backward column pass: dK, dV for a tile of columns (CSC)
# ------------------------------------------------------------------------------------------------------------
@triton.autotune(configs=_configs(), key=["H", "D", "SCORE", "NORM", "BLOCK_R", "PREC"])
@triton.jit
def _bwd_col_tiled_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, m_ptr, dsum_ptr, dk_ptr, dv_ptr, ccol_ptr,
    seg_col0_ptr, seg_start_ptr, seg_end_ptr, seg_out_ptr, rowidx_ptr, etype_ptr, gate_ptr, bias_ptr,
    scale, lo, hi, eps, n_cols,
    H: tl.constexpr, D: tl.constexpr, H_PAD: tl.constexpr, D_PAD: tl.constexpr, HD_PAD: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, PREC: tl.constexpr,
):
    seg = tl.program_id(0)
    col0 = tl.load(seg_col0_ptr + seg).to(tl.int64)
    start = tl.load(seg_start_ptr + seg).to(tl.int64)
    end = tl.load(seg_end_ptr + seg).to(tl.int64)
    out0 = tl.load(seg_out_ptr + seg).to(tl.int64)

    rr = tl.arange(0, BLOCK_R)
    cols = col0 + rr
    cmask = cols < n_cols
    colend = tl.load(ccol_ptr + cols + 1, mask=cmask, other=0).to(tl.int64)
    colend = tl.where(cmask, colend, end)

    hh = tl.arange(0, H_PAD)
    dd = tl.arange(0, D_PAD)
    hd_mask = (hh[:, None] < H) & (dd[None, :] < D)
    hd_off = hh[:, None] * D + dd[None, :]
    tmask = cmask[:, None, None] & hd_mask[None, :, :]
    coff = cols[:, None, None] * (H * D) + hd_off[None, :, :]
    k2 = tl.reshape(tl.load(k_ptr + coff, mask=tmask, other=0.0).to(tl.float32), [BLOCK_R, HD_PAD])
    v2 = tl.reshape(tl.load(v_ptr + coff, mask=tmask, other=0.0).to(tl.float32), [BLOCK_R, HD_PAD])
    dk_acc = tl.zeros([BLOCK_R, HD_PAD], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_R, HD_PAD], dtype=tl.float32)

    ee = tl.arange(0, BLOCK_E)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + ee
        emask = offs < end
        ecol = tl.sum((offs[:, None] >= colend[None, :]).to(tl.int32), axis=1)
        onehot_t = ((ecol[:, None] == rr[None, :]) & emask[:, None]).to(tl.float32)   # [BE, BR]
        onehot = tl.trans(onehot_t)
        ke = tl.reshape(_dot1h(onehot_t, k2, PREC), [BLOCK_E, H_PAD, D_PAD])
        ve = tl.reshape(_dot1h(onehot_t, v2, PREC), [BLOCK_E, H_PAD, D_PAD])
        rows = tl.load(rowidx_ptr + offs, mask=emask, other=0).to(tl.int64)
        gmask = emask[:, None, None] & hd_mask[None, :, :]
        goff = rows[:, None, None] * (H * D) + hd_off[None, :, :]
        qb = tl.load(q_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        dob = tl.load(do_ptr + goff, mask=gmask, other=0.0).to(tl.float32)
        smask = emask[:, None] & (hh[None, :] < H)
        soff = rows[:, None] * H + hh[None, :]
        lb = tl.load(l_ptr + soff, mask=smask, other=1.0)
        mb = tl.load(m_ptr + soff, mask=smask, other=0.0)
        db = tl.load(dsum_ptr + soff, mask=smask, other=0.0)
        prod = qb * ke * scale
        if SCORE == 1:
            et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
            gb = tl.load(gate_ptr + et[:, None, None] * (H * D) + hd_off[None, :, :], mask=gmask, other=0.0).to(tl.float32)
            s = tl.sum(prod * gb, axis=2)
        else:
            s = tl.sum(prod, axis=2)
            if SCORE == 2:
                et = tl.load(etype_ptr + offs, mask=emask, other=0).to(tl.int64)
                s = s + tl.load(bias_ptr + et[:, None] * H + hh[None, :], mask=smask, other=0.0).to(tl.float32)
        dp = tl.sum(dob * ve, axis=2) - db
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
        dv_acc += _dot1h(onehot, tl.reshape(p_hat[:, :, None] * dob, [BLOCK_E, HD_PAD]), PREC)
        if SCORE == 1:
            dk_acc += _dot1h(onehot, tl.reshape((ds * scale)[:, :, None] * qb * gb, [BLOCK_E, HD_PAD]), PREC)
        else:
            dk_acc += _dot1h(onehot, tl.reshape((ds * scale)[:, :, None] * qb, [BLOCK_E, HD_PAD]), PREC)

    slots = out0 + rr
    ooff = slots[:, None, None] * (H * D) + hd_off[None, :, :]
    tl.store(dk_ptr + ooff, tl.reshape(dk_acc, [BLOCK_R, H_PAD, D_PAD]), mask=tmask)
    tl.store(dv_ptr + ooff, tl.reshape(dv_acc, [BLOCK_R, H_PAD, D_PAD]), mask=tmask)


# ------------------------------------------------------------------------------------------------------------
# wrappers (same contracts as triton_kernel.attn_fwd_partial / attn_bwd_partial / attn_bwd)
# ------------------------------------------------------------------------------------------------------------
def _tables(q, gate, bias, etype):
    dummy = torch.zeros(1, dtype=torch.float32, device=q.device)
    gate_c = gate.contiguous().float() if gate is not None else dummy
    bias_c = bias.contiguous().float() if bias is not None else dummy
    etype_c = etype.contiguous() if etype is not None else torch.zeros(1, dtype=torch.int32, device=q.device)
    return gate_c, bias_c, etype_c


def _dims(h, d):
    h_pad, d_pad = _pow2(h), _pow2(d)
    hd_pad = h_pad * d_pad
    assert hd_pad >= 16, "tl.dot needs h_pad*d_pad >= 16"
    return h_pad, d_pad, hd_pad


def _fold_extras(buf, rows_c, plan, extra, block_r):
    """Add the extra-segment slots of a compact plan onto the caller's accumulator (rows via the compact row list)."""
    nx = plan["n_extra"] * block_r
    if nx == 0:
        return
    er = plan["extra_rows"]
    valid = er >= 0
    rows_g = rows_c[er[valid]].long()
    buf.index_add_(0, rows_g, extra[:nx][valid].reshape(int(valid.sum()), -1).to(buf.dtype))


def attn_fwd_partial(q, k, v, crow, col, etype=None, gate=None, bias=None, *, score="dot", norm="softmax",
                     clamp=(-5.0, 5.0), scale=None, block_r=16, seg_edges=2048, accum=None):
    """Unnormalised partials (acc [n, h, d], l [n, h], m [n, h]) of the block.
    accum=(acc_buf [n, h*d], l_buf [n, h]) (row-strided views allowed; clamp-exp only): compact-accumulate mode —
    only the rows that own an edge are tiled (build_dcsr) and the partials are added in place; returns None."""
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    h_pad, d_pad, hd_pad = _dims(h, d)
    gate_c, bias_c, etype_c = _tables(q, gate, bias, etype)
    meta = dict(H=h, D=d, H_PAD=h_pad, D_PAD=d_pad, HD_PAD=hd_pad, BLOCK_R=block_r,
                SCORE=_SCORE[score], NORM=_NORM[norm], PREC=TILED_PREC)
    if accum is not None:
        assert norm == "clampexp", "compact-accumulate partials need clamp-exp (no running max)"
        acc_buf, l_buf = accum
        assert acc_buf.shape == (n, h * d) and acc_buf.stride(1) == 1 and l_buf.shape == (n, h) and l_buf.stride(1) == 1
        rows_c, crow_c = build_dcsr(crow)
        plan = build_tiles(crow_c, block_r, seg_edges)
        nx = max(plan["n_extra"] * block_r, block_r)
        xacc = torch.empty(nx, h, d, dtype=torch.float32, device=q.device)
        xl = torch.empty(nx, h, dtype=torch.float32, device=q.device)
        if plan["n_seg"] > 0:
            _fwd_tiled_kernel[(plan["n_seg"],)](
                q, k, v, acc_buf, l_buf, l_buf, xacc, xl, crow_c, rows_c,
                plan["seg_row0"], plan["seg_start"], plan["seg_end"], plan["seg_out"],
                col, etype_c, gate_c, bias_c, float(scale), float(clamp[0]), float(clamp[1]),
                rows_c.numel(), plan["n_main"], acc_buf.stride(0), l_buf.stride(0), CA=True, **meta)
        _fold_extras(acc_buf, rows_c, plan, xacc, block_r)
        _fold_extras(l_buf, rows_c, plan, xl, block_r)
        return None
    plan = build_tiles(crow, block_r, seg_edges)
    n_slots = plan["n_main"] + plan["n_extra"] * block_r
    acc = torch.zeros(n_slots, h, d, dtype=torch.float32, device=q.device)
    l = torch.zeros(n_slots, h, dtype=torch.float32, device=q.device)
    m = torch.full((n_slots, h), float("-inf"), dtype=torch.float32, device=q.device)
    if plan["n_seg"] > 0:
        _fwd_tiled_kernel[(plan["n_seg"],)](
            q, k, v, acc, l, m, acc, l, crow, crow, plan["seg_row0"], plan["seg_start"], plan["seg_end"], plan["seg_out"],
            col, etype_c, gate_c, bias_c, float(scale), float(clamp[0]), float(clamp[1]), n, 0, 0, 0, CA=False, **meta)
    nm = plan["n_main"]
    acc_o, m_o = _combine(acc[:nm], acc[nm:], plan["extra_rows"], n, norm, m[:nm], m[nm:])
    l_o, _ = _combine(l[:nm], l[nm:], plan["extra_rows"], n, norm, m[:nm], m[nm:])
    if norm == "softmax":
        m_o = torch.where(torch.isfinite(m_o), m_o, torch.zeros_like(m_o))
    else:
        m_o = torch.zeros_like(l_o)
    return acc_o, l_o, m_o


def _csc(crow, col, etype, n_cols):
    from gt3d.attention.triton_kernel import build_csc
    return build_csc(crow, col, etype, n_cols)


def _bwd_row_tiled(q, k, v, crow, col, etype, gate, bias, o_or_negdl, do, l, m, *, score, norm, clamp, eps, scale,
                   partial, block_r, seg_edges, dq_accum=None):
    """Row pass over the CSR tiles: (dq, dgate, dbias).  For partial=True: o_or_negdl = -dl [n, h], l = ones.
    dq_accum ([n, h, d] or [n, h*d], rows contiguous): compact-accumulate row pass, dq added in place (returned dq is None)."""
    n, h, d = q.shape
    h_pad, d_pad, hd_pad = _dims(h, d)
    gate_c, bias_c, etype_c = _tables(q, gate, bias, etype)
    t = gate_c.shape[0] if (gate is not None and score == "gated") else (bias_c.shape[0] if (bias is not None and score == "biased") else 1)
    dgate_slots = torch.zeros(_NSLOT, t, h, d, dtype=torch.float32, device=q.device)
    dbias_slots = torch.zeros(_NSLOT, t, h, dtype=torch.float32, device=q.device)
    meta = dict(NSLOT=_NSLOT, T=t, H=h, D=d, H_PAD=h_pad, D_PAD=d_pad, HD_PAD=hd_pad, BLOCK_R=block_r,
                SCORE=_SCORE[score], NORM=_NORM[norm], PARTIAL=partial, PREC=TILED_PREC)
    if dq_accum is not None:
        dq_buf = dq_accum.view(n, h * d) if dq_accum.dim() == 3 else dq_accum
        assert dq_buf.shape == (n, h * d) and dq_buf.stride(1) == 1
        rows_c, crow_c = build_dcsr(crow)
        plan = build_tiles(crow_c, block_r, seg_edges)
        xdq = torch.empty(max(plan["n_extra"] * block_r, block_r), h, d, dtype=torch.float32, device=q.device)
        if plan["n_seg"] > 0:
            _bwd_row_tiled_kernel[(plan["n_seg"],)](
                q, k, v, o_or_negdl, do, l, m, dq_buf, xdq, dgate_slots, dbias_slots, crow_c, rows_c,
                plan["seg_row0"], plan["seg_start"], plan["seg_end"], plan["seg_out"], col, etype_c, gate_c, bias_c,
                float(scale), float(clamp[0]), float(clamp[1]), float(eps), rows_c.numel(), plan["n_main"], dq_buf.stride(0),
                CA=True, **meta)
        _fold_extras(dq_buf, rows_c, plan, xdq, block_r)
        dq_o = None
    else:
        plan = build_tiles(crow, block_r, seg_edges)
        n_slots = plan["n_main"] + plan["n_extra"] * block_r
        dq = torch.zeros(n_slots, h, d, dtype=torch.float32, device=q.device)
        if plan["n_seg"] > 0:
            _bwd_row_tiled_kernel[(plan["n_seg"],)](
                q, k, v, o_or_negdl, do, l, m, dq, dq, dgate_slots, dbias_slots, crow, crow,
                plan["seg_row0"], plan["seg_start"], plan["seg_end"], plan["seg_out"], col, etype_c, gate_c, bias_c,
                float(scale), float(clamp[0]), float(clamp[1]), float(eps), n, 0, 0, CA=False, **meta)
        nm = plan["n_main"]
        dq_o, _ = _combine(dq[:nm], dq[nm:], plan["extra_rows"], n, "clampexp")
    dgate = dgate_slots.sum(0) if (gate is not None and score == "gated") else None
    dbias = dbias_slots.sum(0) if (bias is not None and score == "biased") else None
    return dq_o, dgate, dbias


def _bwd_col_tiled(q, k, v, crow, col, etype, gate, bias, do, l, m, dsum, *, score, norm, clamp, eps, scale, block_r, seg_edges):
    """Column pass over the CSC tiles: (dk, dv) [n_cols, h, d] (do = dO or dacc, l = row sums or ones, dsum = D_i or -dl_i)."""
    n, h, d = q.shape
    n_cols = k.shape[0]
    h_pad, d_pad, hd_pad = _dims(h, d)
    gate_c, bias_c, _ = _tables(q, gate, bias, etype)
    ccol, rowidx, et_csc = _csc(crow, col, etype, n_cols)
    cplan = build_tiles(ccol, block_r, seg_edges)
    c_slots = cplan["n_main"] + cplan["n_extra"] * block_r
    dk = torch.zeros(c_slots, h, d, dtype=torch.float32, device=q.device)
    dv = torch.zeros(c_slots, h, d, dtype=torch.float32, device=q.device)
    et_arg = et_csc if et_csc is not None else torch.zeros(1, dtype=torch.int32, device=q.device)
    if cplan["n_seg"] > 0:
        _bwd_col_tiled_kernel[(cplan["n_seg"],)](
            q, k, v, do, l, m, dsum, dk, dv, ccol,
            cplan["seg_row0"], cplan["seg_start"], cplan["seg_end"], cplan["seg_out"], rowidx, et_arg, gate_c, bias_c,
            float(scale), float(clamp[0]), float(clamp[1]), float(eps), n_cols,
            H=h, D=d, H_PAD=h_pad, D_PAD=d_pad, HD_PAD=hd_pad, BLOCK_R=block_r,
            SCORE=_SCORE[score], NORM=_NORM[norm], PREC=TILED_PREC)
    cm = cplan["n_main"]
    dk_o, _ = _combine(dk[:cm], dk[cm:], cplan["extra_rows"], n_cols, "clampexp")
    dv_o, _ = _combine(dv[:cm], dv[cm:], cplan["extra_rows"], n_cols, "clampexp")
    return dk_o, dv_o


def _bwd_common(q, k, v, crow, col, etype, gate, bias, o_or_negdl, do, l, m, dsum, *, score, norm, clamp, eps, scale,
                partial, block_r, seg_edges, dq_accum=None):
    """Row pass (dq, dgate, dbias) + column pass (dk, dv), both tiled."""
    dq_o, dgate, dbias = _bwd_row_tiled(q, k, v, crow, col, etype, gate, bias, o_or_negdl, do, l, m, score=score, norm=norm,
                                        clamp=clamp, eps=eps, scale=scale, partial=partial, block_r=block_r,
                                        seg_edges=seg_edges, dq_accum=dq_accum)
    dk_o, dv_o = _bwd_col_tiled(q, k, v, crow, col, etype, gate, bias, do, l, m, dsum, score=score, norm=norm, clamp=clamp,
                                eps=eps, scale=scale, block_r=block_r, seg_edges=seg_edges)
    return dq_o, dk_o, dv_o, dgate, dbias


def bwd_row_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, neg_dl, *, score="dot", norm="softmax",
                    clamp=(-5.0, 5.0), scale=None, block_r=16, seg_edges=2048, dq_accum=None):
    """Tiled row pass of the partial backward (inputs as prepared by triton_kernel._partial_args: contiguous, m fp32,
    neg_dl = -dl fp32, dacc in its own dtype): (dq or None with dq_accum, dgate, dbias)."""
    d = q.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    ones = torch.ones_like(m)
    return _bwd_row_tiled(q, k, v, crow, col, etype, gate, bias, neg_dl, dacc, ones, m, score=score, norm=norm, clamp=clamp,
                          eps=0.0, scale=scale, partial=True, block_r=block_r, seg_edges=seg_edges, dq_accum=dq_accum)


def bwd_col_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, neg_dl, *, score="dot", norm="softmax",
                    clamp=(-5.0, 5.0), scale=None, block_r=16, seg_edges=2048):
    """Tiled column pass of the partial backward: (dk, dv)."""
    d = q.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    ones = torch.ones_like(m)
    return _bwd_col_tiled(q, k, v, crow, col, etype, gate, bias, dacc, ones, m, neg_dl, score=score, norm=norm, clamp=clamp,
                          eps=0.0, scale=scale, block_r=block_r, seg_edges=seg_edges)


def attn_bwd_partial(q, k, v, crow, col, etype, gate, bias, m, dacc, dl, *, score="dot", norm="softmax",
                     clamp=(-5.0, 5.0), scale=None, block_r=16, seg_edges=2048, dq_accum=None):
    """(dq, dk, dv, dgate, dbias) of the unnormalised partials; dq_accum: compact-accumulate mode (dq += in place,
    returned dq is None), see _bwd_common."""
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dacc = dacc.contiguous()                                                # own dtype (bf16 allowed): the kernels upcast
    neg_dl = (-dl).contiguous().float()
    m = m.contiguous().float()
    ones = torch.ones_like(m)
    # column pass reads (l = 1, m, D_i = -dl_i) per row like the row pass
    return _bwd_common(q, k, v, crow, col, etype, gate, bias, neg_dl, dacc, ones, m, neg_dl, score=score, norm=norm,
                       clamp=clamp, eps=0.0, scale=scale, partial=True, block_r=block_r, seg_edges=seg_edges,
                       dq_accum=dq_accum)


def attn_bwd(q, k, v, crow, col, etype, gate, bias, o, l, m, do, *, score="dot", norm="softmax",
             clamp=(-5.0, 5.0), eps=1e-6, scale=None, block_r=16, seg_edges=2048):
    n, h, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = o.contiguous().float()
    do = do.contiguous().float()
    l = l.contiguous().float()
    m = m.contiguous().float()
    dsum = (do * o).sum(-1)                                                 # D_i [n, h]
    return _bwd_common(q, k, v, crow, col, etype, gate, bias, o, do, l, m, dsum, score=score, norm=norm,
                       clamp=clamp, eps=eps, scale=scale, partial=False, block_r=block_r, seg_edges=seg_edges)
