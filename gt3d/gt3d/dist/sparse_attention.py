"""Distributed masked sparse attention on a Grid2D (rows x columns); DistHeadSparseAttention adds the head axis (Grid3D).

One autograd Function does, per layer:
  forward : all-gather K,V over the column group (column block c), all-gather Q over the row
            group (row block r); local kernel -> unnormalised partials (acc_c, l_c, m_c) of block
            A[r, c]; split-KV combine over the row group (all-reduce max of m for softmax, then a
            reduce-scatter of [acc | l] rescaled to the global max) landing on the owner of each row;
            o = acc / l  (or acc / (l + eps) for Exphormer's clamp-exp).
  backward: all-gather (o, l, m, dO) of the row block, kernel backward with the GLOBAL row
            statistics (exactly what FlashAttention's split-KV backward needs), then
            reduce-scatter dQ over the row group and dK, dV over the column group; the per-type
            gate/bias gradient returned is the rank's partial (summed by the caller's grad all-reduce).
Communication per layer and rank: fwd 2 gathers of [P_d*chunk, H, d] + 1 gather of [P_o*chunk, H, d]
+ 1 reduce-scatter of [P_o*chunk, H*d+H]; bwd 1 gather of [P_o*chunk, 2*H*d+2H] + 3 reduce-scatters.
"""
import torch
import torch.distributed as dist

from gt3d.attention.backend_torch import build_csr
from gt3d.dist.grid2d import Grid2D


def _all_gather(x, group):
    g = dist.get_world_size(group)
    out = x.new_empty((g * x.shape[0],) + tuple(x.shape[1:]))
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter(x, group):
    g = dist.get_world_size(group)
    out = x.new_empty((x.shape[0] // g,) + tuple(x.shape[1:]))
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


class _NoTimers:
    def start(self, *a):
        pass

    def stop(self, *a):
        pass


def _timers():
    from axonn import axonn as ax
    t = ax.get_timers()
    return t if t is not None else _NoTimers()           # AxoNN not initialised (Grid3D / plain torch groups)


def _backend(backend):
    if backend == "triton":
        from gt3d.attention import triton_kernel as tk
        return tk.attn_fwd_partial, tk.attn_bwd, tk.normalize
    from gt3d.attention import backend_torch as bt
    return bt.torch_attn_fwd_partial, bt.torch_attn_bwd, lambda acc, l, norm, eps, dtype=None: bt.torch_normalize(acc, l, norm, eps)


class _DistAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_loc, k_loc, v_loc, gate, module):
        m_ = module
        g = m_.grid
        fwd_partial, _, normalize = _backend(m_.backend)
        cdt = m_.comm_dtype or q_loc.dtype                    # e.g. bf16 halves the gather traffic; kernels upcast
        k_blk = _all_gather(k_loc.to(cdt), g.col_group)
        v_blk = _all_gather(v_loc.to(cdt), g.col_group)
        q_blk = _all_gather(q_loc.to(cdt), g.row_group)
        col = m_.col if m_.backend == "triton" else m_.col.long()
        et = m_.etype if m_.backend == "triton" else (m_.etype.long() if m_.etype is not None else None)
        gate_c = gate.to(torch.float32) if gate is not None else None
        if m_.backend != "triton":                           # torch backend computes in the input dtype
            q_blk, k_blk, v_blk = q_blk.to(q_loc.dtype), k_blk.to(q_loc.dtype), v_blk.to(q_loc.dtype)
            gate_c = gate.to(q_loc.dtype) if gate is not None else None
        acc, l, m = fwd_partial(q_blk, k_blk, v_blk, m_.crow, col, et, gate_c, None,
                                score=m_.score, norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
        sdt = torch.float64 if q_loc.dtype == torch.float64 else torch.float32   # statistics dtype
        acc, l, m = acc.to(sdt), l.to(sdt), m.to(sdt)
        if m_.norm == "softmax":
            m_glob = m.clone()
            dist.all_reduce(m_glob, op=dist.ReduceOp.MAX, group=g.row_group)
            w = torch.exp(m - m_glob)
            acc = acc * w.unsqueeze(-1)
            l = l * w
            m = m_glob
        n_rows, h, d = acc.shape
        packed = _reduce_scatter(torch.cat([acc.reshape(n_rows, -1), l], dim=1), g.row_group)
        acc_o = packed[:, : h * d].view(-1, h, d)
        l_o = packed[:, h * d:].contiguous()
        m_o = m[g.c * g.chunk:(g.c + 1) * g.chunk].contiguous()            # my rows inside the row block
        o_loc = normalize(acc_o, l_o, m_.norm, m_.eps).to(q_loc.dtype)
        ctx.module = m_
        ctx.save_for_backward(q_blk, k_blk, v_blk, o_loc, l_o, m_o,
                              gate_c if gate_c is not None else torch.zeros(0, device=q_loc.device))
        ctx.has_gate = gate is not None
        ctx.gate_dtype = gate.dtype if gate is not None else None
        ctx.in_dtype = q_loc.dtype
        return o_loc

    @staticmethod
    def backward(ctx, do_loc):
        m_ = ctx.module
        g = m_.grid
        _, bwd, _ = _backend(m_.backend)
        q_blk, k_blk, v_blk, o_loc, l_o, m_o, gate_c = ctx.saved_tensors
        h, d = q_blk.shape[1], q_blk.shape[2]
        # gather (o, dO, l, m) of the whole row block in one collective
        sdt = l_o.dtype
        cdt = m_.comm_dtype or ctx.in_dtype
        if cdt == sdt:                                        # one collective for (o, dO, l, m)
            packed = torch.cat([o_loc.reshape(o_loc.shape[0], -1).to(sdt), do_loc.reshape(do_loc.shape[0], -1).to(sdt),
                                l_o, m_o], dim=1)
            packed = _all_gather(packed, g.row_group)
            o_blk = packed[:, : h * d].view(-1, h, d)
            do_blk = packed[:, h * d: 2 * h * d].view(-1, h, d)
            l_blk = packed[:, 2 * h * d: 2 * h * d + h].contiguous()
            m_blk = packed[:, 2 * h * d + h:].contiguous()
        else:                                                 # low-precision (o, dO); statistics stay in sdt
            odo = _all_gather(torch.cat([o_loc.reshape(o_loc.shape[0], -1), do_loc.reshape(do_loc.shape[0], -1)], dim=1).to(cdt), g.row_group)
            o_blk = odo[:, : h * d].view(-1, h, d)
            do_blk = odo[:, h * d:].view(-1, h, d)
            lm = _all_gather(torch.cat([l_o, m_o], dim=1), g.row_group)
            l_blk = lm[:, :h].contiguous()
            m_blk = lm[:, h:].contiguous()
        col = m_.col if m_.backend == "triton" else m_.col.long()
        et = m_.etype if m_.backend == "triton" else (m_.etype.long() if m_.etype is not None else None)
        gate_arg = gate_c if ctx.has_gate else None
        dq_blk, dk_blk, dv_blk, dgate, _ = bwd(q_blk, k_blk, v_blk, m_.crow, col, et, gate_arg, None,
                                               o_blk, l_blk, m_blk, do_blk, score=m_.score, norm=m_.norm,
                                               clamp=m_.clamp, eps=m_.eps, scale=m_.scale)
        rdt = ctx.in_dtype                                    # partial sums are reduced in the model's dtype
        dq_loc = _reduce_scatter(dq_blk.to(rdt), g.row_group)
        dk_loc = _reduce_scatter(dk_blk.to(rdt), g.col_group)
        dv_loc = _reduce_scatter(dv_blk.to(rdt), g.col_group)
        # dgate is this rank's PARTIAL (edges of its block); like every replicated parameter it is summed
        # over ranks by the caller's gradient all-reduce.
        if dgate is not None:
            dgate = dgate.to(ctx.gate_dtype)
        return dq_loc, dk_loc, dv_loc, dgate, None


class AttentionBlock:
    """CSR of the attention block A[r, c] (row-local dst, col-local src, edge types), built once
    and shared by all attention layers of a model."""

    def __init__(self, grid: Grid2D, block_edge_index, block_edge_type):
        n_rows = grid.n_row_block
        crow, col, et, _ = build_csr(block_edge_index, n_rows, block_edge_type)
        assert col.numel() < 2 ** 31, "local block too large for int32 edge indices"
        self.crow, self.col = crow.to(torch.int32).contiguous(), col.to(torch.int32)      # int32 crow: the kernels cast
        self.etype = et.to(torch.int32) if et is not None else None
        self.num_edges = int(col.numel())

    @classmethod
    def from_csr(cls, crow, col, etype):
        """An AttentionBlock from an existing CSR (e.g. the compact tile of a sparse-communication plan)."""
        self = cls.__new__(cls)
        assert col.numel() < 2 ** 31, "local block too large for int32 edge indices"
        self.crow, self.col = crow.to(torch.int32).contiguous(), col.to(torch.int32).contiguous()
        self.etype = etype.to(torch.int32).contiguous() if etype is not None else None
        self.num_edges = int(col.numel())
        return self

    def compacted(self, rows_need, col_remap):
        """The tile over the row set rows_need (must contain every non-empty row) with the columns remapped through
        col_remap (block-local column -> compact column): the CSR the kernels run on when the row block / column block
        are gathered sparsely (gt3d.dist.sparse_comm)."""
        from gt3d.dist.sparse_comm import compact_csr
        crow_c, col_c = compact_csr(self.crow, rows_need, self.col, col_remap)
        return AttentionBlock.from_csr(crow_c, col_c, self.etype)

    def nbytes(self, seen=None):
        """Bytes of the persistent structures incl. the plans the kernels cache on the CSR / CSC tensors."""
        return structure_tensor_bytes([self.crow, self.col, self.etype], seen)


def structure_tensor_bytes(tensors, seen=None):
    """Bytes of `tensors` plus the plans the kernels cache on them (`_gt3d_tiles`, `_gt3d_csc`, `_gt3d_seg_plans`,
    recursively); storages already in `seen` (set of data_ptr) are not counted twice."""
    seen = set() if seen is None else seen
    total = 0

    def walk(obj, depth=0):
        nonlocal total
        if torch.is_tensor(obj):
            if obj.data_ptr() not in seen:
                seen.add(obj.data_ptr()); total += obj.numel() * obj.element_size()
            if depth < 3:
                for attr in ("_gt3d_tiles", "_gt3d_csc", "_gt3d_seg_plans"):
                    walk(getattr(obj, attr, None), depth + 1)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth)

    walk(list(tensors))
    return total


class DistMaskedSparseAttention(torch.nn.Module):
    """Attention over the local block A[r, c]; inputs/outputs are the owned node chunk."""

    def __init__(self, grid: Grid2D, block, num_heads, head_dim,
                 backend="triton", score="gated", norm="clampexp", clamp=(-5.0, 5.0), eps=1e-6, comm_dtype=None):
        super().__init__()
        if not isinstance(block, AttentionBlock):             # (edge_index, edge_type) tuple for convenience
            block = AttentionBlock(grid, *block)
        self.grid = grid
        self.comm_dtype = comm_dtype                          # None: model dtype; torch.bfloat16 to halve gather traffic
        self.h, self.d = num_heads, head_dim
        self.backend, self.score, self.norm, self.clamp, self.eps = backend, score, norm, clamp, eps
        self.scale = 1.0 / (head_dim ** 0.5)
        self.crow, self.col, self.etype = block.crow, block.col, block.etype

    @torch.compiler.disable
    def forward(self, q_local, k_local, v_local, gate=None):
        return _DistAttention.apply(q_local.contiguous(), k_local.contiguous(), v_local.contiguous(), gate, self)


# =============================================================================================
# Fused-communication variant: the layer gathers its INPUT once per block; the attention Function
# below consumes already-gathered q (row block), k, v (column block) and returns the UNNORMALISED
# partials (acc, l, m) of the local block.  Its backward receives (dacc, dl) from autograd (the
# reduce-scatter's backward is an all-gather), so no global statistics and no (o, dO) gather are
# needed; dq/dk/dv flow back into the gather wrappers whose backward reduce-scatters them.
# =============================================================================================
class _AttnPartial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_blk, k_blk, v_blk, gate, bias, module):
        m_ = module
        fwd_partial, _, _ = _backend(m_.backend)
        col = m_.col if m_.backend == "triton" else m_.col.long()
        et = m_.etype if m_.backend == "triton" else (m_.etype.long() if m_.etype is not None else None)
        tdt = torch.float32 if m_.backend == "triton" else q_blk.dtype
        gate_c = gate.to(tdt) if gate is not None else None
        bias_c = bias.to(tdt) if bias is not None else None                  # [T, h] per-type additive score bias
        t = _timers(); t.start("attn fwd kernel")
        acc, l, m = fwd_partial(q_blk, k_blk, v_blk, m_.crow, col, et, gate_c, bias_c,
                                score=m_.score, norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
        t.stop("attn fwd kernel")
        ctx.module = m_
        zero = torch.zeros(0, device=q_blk.device)
        ctx.save_for_backward(q_blk, k_blk, v_blk, m, gate_c if gate_c is not None else zero, bias_c if bias_c is not None else zero)
        ctx.has_gate, ctx.has_bias = gate is not None, bias is not None
        ctx.gate_dtype = gate.dtype if gate is not None else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return acc, l, m

    @staticmethod
    def backward(ctx, dacc, dl, dm):
        m_ = ctx.module
        q_blk, k_blk, v_blk, m, gate_c, bias_c = ctx.saved_tensors
        col = m_.col if m_.backend == "triton" else m_.col.long()
        et = m_.etype if m_.backend == "triton" else (m_.etype.long() if m_.etype is not None else None)
        gate_arg = gate_c if ctx.has_gate else None
        bias_arg = bias_c if ctx.has_bias else None
        if m_.backend == "triton":
            from gt3d.attention.triton_kernel import attn_bwd_partial as bwd
        else:
            from gt3d.attention.backend_torch import torch_attn_bwd_partial as bwd
        if k_blk.dtype == torch.bfloat16 and m_.backend == "triton":
            dacc = dacc.to(torch.bfloat16)                 # the column pass reads dacc per edge: halve it like K / V
        t = _timers(); t.start("attn bwd kernel")
        dq, dk, dv, dgate, dbias = bwd(q_blk, k_blk, v_blk, m_.crow, col, et, gate_arg, bias_arg, m, dacc, dl,
                                       score=m_.score, norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
        t.stop("attn bwd kernel")
        return (dq.to(q_blk.dtype), dk.to(k_blk.dtype), dv.to(v_blk.dtype),
                dgate.to(ctx.gate_dtype) if (dgate is not None and ctx.has_gate) else None,
                dbias.to(ctx.bias_dtype) if (dbias is not None and ctx.has_bias) else None, None)


class DistAttentionPartial(torch.nn.Module):
    """Unnormalised block attention on already-gathered q (row block) / k, v (column block)."""

    def __init__(self, grid: Grid2D, block, num_heads, head_dim, backend="triton", score="gated",
                 norm="clampexp", clamp=(-5.0, 5.0)):
        super().__init__()
        if not isinstance(block, AttentionBlock):
            block = AttentionBlock(grid, *block)
        self.grid = grid
        self.h, self.d = num_heads, head_dim
        self.backend, self.score, self.norm, self.clamp = backend, score, norm, clamp
        self.scale = 1.0 / (head_dim ** 0.5)
        self.crow, self.col, self.etype = block.crow, block.col, block.etype

    @torch.compiler.disable
    def forward(self, q_blk, k_blk, v_blk, gate=None, bias=None):
        return _AttnPartial.apply(q_blk.contiguous(), k_blk.contiguous(), v_blk.contiguous(), gate, bias, self)


@torch.compiler.disable
def combine_partials(acc, l, m, grid: Grid2D, norm: str, eps: float, extra=None, rs_dtype=None, plan=None):
    """Split-KV combine of the local partials over the row group, landing on the owner of every row.
    `extra` ([n_rows, F] row-block partial sums, e.g. the GCN aggregation) rides in the same
    reduce-scatter.  rs_dtype (e.g. bfloat16) reduces the payload in low precision (NCCL accumulates in
    that dtype; with P_o <= 8 partial sums the relative error is ~1e-3).  Returns (o_owned, extra_owned).
    plan: a SparseRowPlan of the row group (sparse communication): the partials are given for the plan's
    compact rows only (the non-empty rows of the tile) and go back to their owners with an all-to-all-v."""
    from gt3d.dist.grid2d import reduce_scatter_rows
    if norm == "softmax":
        if plan is None:
            m_glob = m.detach().clone()
        else:                                                  # same shape on every rank: scatter into the row block
            m_glob = torch.full((plan.g * plan.rows_per_rank, m.shape[1]), float("-inf"), dtype=m.dtype, device=m.device)
            m_glob[plan.need_local.long()] = m.detach()
        dist.all_reduce(m_glob, op=dist.ReduceOp.MAX, group=grid.row_group)
        if plan is not None:
            m_glob = m_glob[plan.need_local.long()]
        w = torch.exp(m.detach() - m_glob)                     # rescale to the global max (no grad through m)
        acc = acc * w.unsqueeze(-1)
        l = l * w
    n_rows, h, d = acc.shape
    parts = [acc.reshape(n_rows, -1), l]
    width = h * d + h
    if extra is not None:
        parts.append(extra.to(acc.dtype))
        width += extra.shape[1]
    if plan is not None:
        # sparse reduce-scatter: pad the payload rows to a multiple of 16 bytes here (free inside the cat) so that the
        # plan need not copy the whole row block to align it (see SparseRowPlan._aligned)
        wire = rs_dtype if rs_dtype is not None else acc.dtype
        pad = (-width) % (16 // torch.empty(0, dtype=wire).element_size())
        if pad:
            parts.append(acc.new_zeros(n_rows, pad))
    packed = torch.cat(parts, dim=1)
    out_dtype = packed.dtype
    if rs_dtype is not None and rs_dtype != packed.dtype:
        packed = packed.to(rs_dtype)
    if plan is None:
        packed = reduce_scatter_rows(packed, grid.row_group).to(out_dtype)
    else:
        from gt3d.dist.sparse_comm import sparse_reduce_scatter
        packed = sparse_reduce_scatter(packed, plan).to(out_dtype)
    acc_o = packed[:, : h * d].view(-1, h, d)
    l_o = packed[:, h * d: h * d + h]
    if norm == "softmax":
        o = acc_o / torch.where(l_o > 0, l_o, torch.ones_like(l_o)).unsqueeze(-1)
    else:
        o = acc_o / (l_o + eps).unsqueeze(-1)
    extra_o = packed[:, h * d + h: width] if extra is not None else None
    return o, extra_o


def split_block_by_column_subblocks(edge_index_local, n_sub, chunk, p_rows, etype=None, weight=None):
    """Partition a local block (col-local src, row-local dst) by column sub-block s = (src % chunk) // cs
    (cs = chunk / n_sub), remapping src to the sub-block's own column-local index
    (r' * cs + (local_row - s*cs)), matching the row order of the gathered sub-block."""
    assert chunk % n_sub == 0, (chunk, n_sub)
    cs = chunk // n_sub
    src, dst = edge_index_local[0], edge_index_local[1]
    rnk, loc = src // chunk, src % chunk
    sub = loc // cs
    out = []
    for s in range(n_sub):
        m = sub == s
        new_src = rnk[m] * cs + (loc[m] - s * cs)
        out.append((torch.stack([new_src, dst[m]]), etype[m] if etype is not None else None,
                    weight[m] if weight is not None else None))
    return out


# =============================================================================================
# Head axis (Grid3D): DeepSpeed-Ulysses-style head parallelism on top of the 2D block grid.
# =============================================================================================
class DistHeadSparseAttention(torch.nn.Module):
    """Attention over the local block A[r, c] restricted to head group h of a Grid3D.  q, k, v are the owned chunk
    with ALL heads [chunk, H, d] (projected by the owner: no redundant projections); o is the owned chunk with all
    heads.  Per layer and rank, forward:
      q, k, v : all-to-all rows <-> heads inside the head group (P_i ranks, Ulysses), then all-gather of the head
                slice over the row group (Q, P_o ranks) / the column group (K, V, P_d ranks);
      kernel  : block partials (acc, l, m) for the H/P_i heads of the group;
      combine : split-KV reduce-scatter over the row group, normalisation, all-to-all heads -> rows.
    Every collective is an autograd Function (all-gather <-> reduce-scatter, a2a <-> inverse a2a), so the backward
    is the exact mirror and needs no global statistics.  (P_d, P_o, 1) = the 2D grid with projected gathers;
    (1, 1, P) = pure head parallelism (DeepSpeed-Ulysses / TorchGT / DAC'26 GP-A2A): whole graph per rank, P <= H.
    comm_dtype (e.g. bfloat16) is the wire dtype of the a2a / gathers (and of their backward reduce-scatters);
    rs_dtype the wire dtype of the combine; no_comm replaces every collective by a local op of the same shape."""

    def __init__(self, grid, block, num_heads, head_dim, backend="triton", score="gated", norm="clampexp",
                 clamp=(-5.0, 5.0), eps=1e-6, comm_dtype=None, rs_dtype=None, no_comm=False):
        super().__init__()
        assert num_heads % grid.p_heads == 0, (num_heads, grid.p_heads)
        if not isinstance(block, AttentionBlock):
            block = AttentionBlock(grid, *block)
        self.grid = grid
        self.h, self.d, self.hg = num_heads, head_dim, num_heads // grid.p_heads
        self.norm, self.eps = norm, eps
        self.comm_dtype, self.rs_dtype, self.no_comm = comm_dtype, rs_dtype, no_comm
        self.partial = DistAttentionPartial(grid, block, self.hg, head_dim, backend=backend, score=score,
                                            norm=norm, clamp=clamp)

    def _to_block(self, t, group):
        """owned chunk, all heads [chunk, H, d] -> gathered block of my head group [G*P_i*chunk, H/P_i, d]."""
        from gt3d.dist.grid2d import all_gather_rows
        from gt3d.dist.grid3d import rows_to_heads
        g = self.grid
        wire = t.to(self.comm_dtype) if self.comm_dtype is not None else t
        if self.no_comm:
            blk = wire[:, g.h * self.hg:(g.h + 1) * self.hg].repeat(g.p_heads * dist.get_world_size(group), 1, 1)
        else:
            blk = all_gather_rows(rows_to_heads(wire, g.head_group, g.p_heads), group)
        return blk.to(t.dtype)

    @torch.compiler.disable
    def forward(self, q_loc, k_loc, v_loc, gate=None):
        from gt3d.dist.grid3d import heads_to_rows
        g = self.grid
        t = _timers()
        t.start("head: a2a + gather q/k/v")
        q_blk = self._to_block(q_loc.contiguous(), g.row_group)
        k_blk = self._to_block(k_loc.contiguous(), g.col_group)
        v_blk = self._to_block(v_loc.contiguous(), g.col_group)
        t.stop("head: a2a + gather q/k/v")
        gate_h = gate[:, g.h * self.hg:(g.h + 1) * self.hg].contiguous() if gate is not None else None
        t.start("attention partial")
        acc, l, m = self.partial(q_blk, k_blk, v_blk, gate_h)
        t.stop("attention partial")
        if self.no_comm:                                                 # local stand-ins (timing only, clampexp)
            n_rows, hg, d = acc.shape
            g_row = dist.get_world_size(g.row_group)
            packed = torch.cat([acc.reshape(n_rows, -1), l], dim=1)
            if self.rs_dtype is not None:
                packed = packed.to(self.rs_dtype)
            packed = packed.view(g_row, n_rows // g_row, -1).sum(0).to(acc.dtype)
            o_h = packed[:, : hg * d].view(-1, hg, d) / (packed[:, hg * d:] + self.eps).unsqueeze(-1)
            chunk = o_h.shape[0] // g.p_heads
            return o_h.view(g.p_heads, chunk, hg, d).transpose(0, 1).reshape(chunk, self.h, d).to(q_loc.dtype)
        t.start("reduce-scatter")
        o_h, _ = combine_partials(acc, l, m, g, self.norm, self.eps, extra=None, rs_dtype=self.rs_dtype)
        t.stop("reduce-scatter")
        t.start("head: a2a back")
        wire = o_h.to(self.comm_dtype) if self.comm_dtype is not None else o_h
        out = heads_to_rows(wire, g.head_group, g.p_heads).to(q_loc.dtype)
        t.stop("head: a2a back")
        return out
