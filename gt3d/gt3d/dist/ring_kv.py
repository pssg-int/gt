"""Ring attention with K/V rotation (Ring Attention as published) for structure-masked sparse attention; attention only.

Layout: a (P, 1) Grid2D = pure row (query) partition; rank r owns node chunk r.  gt3d/dist/ring.py rotates the LAYER
INPUT x around the ring and projects K, V of every chunk at every hop (P projection GEMMs per rank and layer, and the
matching dW_k / dW_v / dx GEMMs per hop in the backward).  This variant is Ring Attention (Liu et al. 2023) as
published: every rank projects K, V of its own chunk ONCE (outside this module, by autograd), and the packed [K | V]
chunk ([2, chunk, H*d] in the comm dtype) travels around the ring of the P ranks (NCCL P2P, one batched isend/irecv
per hop).  At step s rank r holds the K/V of chunk c = (r - s) mod P and runs the fused CSR kernel on the sub-block
A[r, c] (the edges of its rows whose sources lie in chunk c) while the chunk is already on its way to rank r + 1.  The
unnormalised clamp-exp partials (acc, l) of the P sub-blocks add up (no rescaling); o = acc / (l + eps) is formed by
the module, so the backward receives (dacc, dl) from autograd and needs no global row statistics.

Backward (Ring Attention's): only the owned q, k, v are saved.  K/V travel again, dQ accumulates locally, and the
[dK | dV] of every chunk travels WITH the chunk as an accumulator to which every rank adds its sub-block's
contribution; after P hops it lands on the owner, whose projection backward (autograd) turns it into dW_k, dW_v, dx.

Traffic per rank and layer (chunk = N/P rows, hidden = H*d, b bytes per element): forward (P-1) hops of
2 * chunk * H*d * b; backward (P-1) hops of K/V again + P hops of the dK/dV accumulator (2 * chunk * H*d * b each).
With hidden = H*d this is exactly twice the bytes of the x-rotating ring (ring.py); in exchange there is no
redundant per-hop projection (the K/V GEMM runs once per chunk instead of once per hop, and the backward has no
per-hop GEMMs at all).  Memory O(N/P): one double-buffered [2, chunk, H*d] chunk plus the accumulator.  The Triton
kernels read K/V in the comm dtype directly (they upcast per edge), so there is no per-hop conversion pass either.

P2P safety: as in ring.py, every step posts one batched {isend to next, irecv from prev} per travelling buffer and
every rank issues the same sequence of batched groups.

Bidirectional ring (`bidir=True`, bench strategy ring_kv2): a cross-node hop is one P2P transfer over the one NIC that
links the two ring neighbours, and the links are full duplex, so a unidirectional ring uses half of each cross-node
link.  Here every chunk's rows are split in two halves: half A (rows [0, h_a)) travels clockwise (to rank + 1), half B
(rows [h_a, chunk)) counter-clockwise (to rank - 1); at step s rank r holds half A of chunk (r - s) mod P and half B of
chunk (r + s) mod P, concatenated into ONE [2, chunk, H d] buffer in chunk-local row order, and runs ONE kernel per step
on the sub-block of its edges whose sources are in {chunk r - s, half A} or {chunk r + s, half B} (chunk-local source
index; every edge belongs to exactly one step).  Same number of kernel passes and the same bytes per rank as the
unidirectional ring, but each hop puts half of them on each direction of every link.  Both directions are posted in
one batched NCCL group per step (one kernel, so they overlap); the backward rotates the two halves and their two
dK/dV accumulators the same way (A's accumulator clockwise, B's counter-clockwise).
"""
import torch
import torch.distributed as dist

from gt3d.dist.ring import RingComm, split_block_by_source_chunk
from gt3d.dist.sparse_attention import AttentionBlock, _timers


def _kernels(backend):
    """(fwd_partial, bwd_partial, col-cast, etype-cast) of the backend."""
    if backend == "triton":
        from gt3d.attention.triton_kernel import attn_fwd_partial, attn_bwd_partial
        return attn_fwd_partial, attn_bwd_partial, (lambda t: t), (lambda t: t)
    from gt3d.attention.backend_torch import torch_attn_fwd_partial, torch_attn_bwd_partial
    return torch_attn_fwd_partial, torch_attn_bwd_partial, (lambda t: t.long()), (lambda t: None if t is None else t.long())


def split_block_bidir(edge_index_local, p, chunk, rank, h_a, etype=None):
    """Split the local block of a (P, 1) grid (src = global id, dst = row-local) into the P step sub-blocks of the
    bidirectional ring of `rank`: an edge with source in chunk c, chunk-local row loc, belongs to step
    (rank - c) mod P if loc < h_a (half A, clockwise) else (c - rank) mod P (half B, counter-clockwise); the source
    is remapped to its chunk-local index loc, which is where that row sits in the step's concatenated buffer."""
    src, dst = edge_index_local[0], edge_index_local[1]
    c, loc = src // chunk, src % chunk
    s_of = torch.where(loc < h_a, (rank - c) % p, (c - rank) % p)
    out = []
    for s in range(p):
        m = s_of == s
        out.append((torch.stack([loc[m], dst[m]]), etype[m] if etype is not None else None))
    return out


class _RingAttnKV(torch.autograd.Function):
    """Owned rows' unnormalised partials (acc [chunk, H, d], l [chunk, H]) summed over the P ring steps.
    Saved for backward: the owned q, k, v and the gate only."""

    @staticmethod
    def _pack(k, v, cdt):
        n = k.shape[0]
        return torch.stack([k.reshape(n, -1), v.reshape(n, -1)]).to(cdt).contiguous()      # [2, chunk, H*d]

    @staticmethod
    def forward(ctx, q, k, v, gate, module):
        m_ = module
        ring, P = m_.ring, m_.ring.size
        h, d = m_.h, m_.d
        n = q.shape[0]
        fwd, _, ci, ei = _kernels(m_.backend)
        cdt = m_.comm_dtype or q.dtype                                     # wire dtype of the travelling K/V
        kdt = cdt if m_.backend == "triton" else q.dtype                   # dtype the kernel consumes (Triton upcasts)
        pdt = torch.float64 if q.dtype == torch.float64 else torch.float32  # partials / statistics dtype
        gate_c = gate.to(torch.float32 if m_.backend == "triton" else q.dtype) if gate is not None else None
        t = _timers()
        with torch.no_grad():
            buf = _RingAttnKV._pack(k, v, cdt)                             # the chunk currently held (my own at step 0)
            nbuf = torch.empty_like(buf) if P > 1 else None
            acc = torch.zeros(n, h, d, dtype=pdt, device=q.device)
            l = torch.zeros(n, h, dtype=pdt, device=q.device)
            works = None
            for s in range(P):
                c = (ring.rank - s) % P                                     # source chunk held at this step
                if s < P - 1 and not m_.no_comm:
                    works = ring.send_recv(buf, nbuf)                       # hop s: overlaps this step's compute
                t.start("ring_kv fwd: compute")
                blk = m_.attns[c]
                if blk.col.numel() > 0:
                    kv = buf if buf.dtype == kdt else buf.to(kdt)
                    acc_s, l_s, _ = fwd(q, kv[0].view(n, h, d), kv[1].view(n, h, d), blk.crow, ci(blk.col), ei(blk.etype),
                                        gate_c, None, score=m_.score, norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
                    acc += acc_s.to(pdt)
                    l += l_s.to(pdt)
                    del acc_s, l_s, kv
                t.stop("ring_kv fwd: compute")
                if s < P - 1 and not m_.no_comm:
                    t.start("ring_kv fwd: wait")
                    ring.wait(works)
                    t.stop("ring_kv fwd: wait")
                    buf, nbuf = nbuf, buf                                   # received chunk becomes current
            del buf, nbuf
        ctx.module = m_
        ctx.save_for_backward(q, k, v, gate_c if gate_c is not None else torch.zeros(0, device=q.device))
        ctx.has_gate = gate is not None
        ctx.gate_dtype = gate.dtype if gate is not None else None
        ctx.pdt = pdt
        return acc, l

    @staticmethod
    def backward(ctx, dacc, dl):
        q, k, v, gate_c = ctx.saved_tensors
        m_ = ctx.module
        ring, P = m_.ring, m_.ring.size
        h, d = m_.h, m_.d
        hd = h * d
        n = q.shape[0]
        _, bwd, ci, ei = _kernels(m_.backend)
        cdt = m_.comm_dtype or q.dtype
        kdt = cdt if m_.backend == "triton" else q.dtype
        pdt = ctx.pdt
        gate_arg = gate_c if ctx.has_gate else None
        t = _timers()
        with torch.no_grad():
            dacc = dacc.contiguous().to(pdt)
            dl = dl.contiguous().to(pdt)
            m0 = torch.zeros_like(dl)                                       # clamp-exp partials have m = 0
            dq = torch.zeros(n, h, d, dtype=pdt, device=q.device)
            dgate = None
            buf = _RingAttnKV._pack(k, v, cdt)
            nbuf = torch.empty_like(buf) if P > 1 else None
            dkv_in = None                                                   # accumulator arriving from prev
            works_kv = works_dkv = None
            dkv_total = None
            for s in range(P):
                c = (ring.rank - s) % P
                if s < P - 1 and not m_.no_comm:
                    works_kv = ring.send_recv(buf, nbuf)                    # K/V for step s+1
                t.start("ring_kv bwd: compute")
                blk = m_.attns[c]
                if blk.col.numel() > 0:
                    kv = buf if buf.dtype == kdt else buf.to(kdt)
                    dq_s, dk_s, dv_s, dgate_s, _ = bwd(q, kv[0].view(n, h, d), kv[1].view(n, h, d), blk.crow, ci(blk.col),
                                                       ei(blk.etype), gate_arg, None, m0, dacc, dl, score=m_.score,
                                                       norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
                    dq += dq_s.to(pdt)
                    if dgate_s is not None:
                        dgate = dgate_s if dgate is None else dgate + dgate_s
                    dkv_s = torch.stack([dk_s.reshape(n, hd), dv_s.reshape(n, hd)]).to(pdt)   # my contribution to chunk c's dK/dV
                    del dq_s, dk_s, dv_s, kv
                else:
                    dkv_s = torch.zeros(2, n, hd, dtype=pdt, device=q.device)
                t.stop("ring_kv bwd: compute")
                if P == 1 or m_.no_comm:
                    dkv_total = dkv_s if dkv_total is None else dkv_total + dkv_s
                    continue
                if s > 0:
                    t.start("ring_kv bwd: wait dkv")
                    ring.wait(works_dkv)                                    # chunk c's accumulator from prev (its step s-1)
                    t.stop("ring_kv bwd: wait dkv")
                    dkv_s += dkv_in.to(pdt)
                dkv_send = dkv_s.to(cdt).contiguous()                       # accumulate in the comm dtype (as a reduce-scatter does)
                dkv_in = torch.empty_like(dkv_send)
                works_dkv = ring.send_recv(dkv_send, dkv_in)                # hop s of the accumulator (P hops in total)
                del dkv_s
                if s < P - 1:
                    t.start("ring_kv bwd: wait kv")
                    ring.wait(works_kv)
                    t.stop("ring_kv bwd: wait kv")
                    buf, nbuf = nbuf, buf
            if P > 1 and not m_.no_comm:
                t.start("ring_kv bwd: wait dkv")
                ring.wait(works_dkv)                                        # my own chunk's accumulator, complete
                t.stop("ring_kv bwd: wait dkv")
                dkv = dkv_in.to(pdt)
            else:
                dkv = dkv_total
            del buf, nbuf
        return (dq.to(q.dtype), dkv[0].view(n, h, d).to(k.dtype), dkv[1].view(n, h, d).to(v.dtype),
                dgate.to(ctx.gate_dtype) if dgate is not None else None, None)


class _RingAttnKV2(torch.autograd.Function):
    """Bidirectional ring (module docstring): half A of the K/V chunk travels clockwise, half B counter-clockwise;
    one kernel per step on the concatenated [2, chunk, H d] buffer and the step's sub-block (attns[s])."""

    @staticmethod
    def _pack_halves(k, v, h_a, cdt):
        n = k.shape[0]
        kk, vv = k.reshape(n, -1), v.reshape(n, -1)
        return (torch.stack([kk[:h_a], vv[:h_a]]).to(cdt).contiguous(),         # [2, h_a, H*d]
                torch.stack([kk[h_a:], vv[h_a:]]).to(cdt).contiguous())         # [2, chunk - h_a, H*d]

    @staticmethod
    def _send_recv2(ring, send_a, recv_a, send_b, recv_b):
        """One batched group: A to next / A' from prev, B to prev / B' from next (same op order on every rank)."""
        import torch.distributed as dist
        for t in (send_a, recv_a, send_b, recv_b):
            assert t.is_contiguous()
        ops = [dist.P2POp(dist.isend, send_a, ring.next, group=ring.group),
               dist.P2POp(dist.irecv, recv_a, ring.prev, group=ring.group),
               dist.P2POp(dist.isend, send_b, ring.prev, group=ring.group),
               dist.P2POp(dist.irecv, recv_b, ring.next, group=ring.group)]
        return dist.batch_isend_irecv(ops)

    @staticmethod
    def forward(ctx, q, k, v, gate, module):
        m_ = module
        ring, P = m_.ring, m_.ring.size
        h, d = m_.h, m_.d
        n = q.shape[0]
        h_a = m_.h_a
        fwd, _, ci, ei = _kernels(m_.backend)
        cdt = m_.comm_dtype or q.dtype
        kdt = cdt if m_.backend == "triton" else q.dtype
        pdt = torch.float64 if q.dtype == torch.float64 else torch.float32
        gate_c = gate.to(torch.float32 if m_.backend == "triton" else q.dtype) if gate is not None else None
        t = _timers()
        with torch.no_grad():
            buf_a, buf_b = _RingAttnKV2._pack_halves(k, v, h_a, cdt)
            nbuf_a, nbuf_b = (torch.empty_like(buf_a), torch.empty_like(buf_b)) if P > 1 else (None, None)
            acc = torch.zeros(n, h, d, dtype=pdt, device=q.device)
            l = torch.zeros(n, h, dtype=pdt, device=q.device)
            works = None
            for s in range(P):
                if s < P - 1 and not m_.no_comm:
                    works = _RingAttnKV2._send_recv2(ring, buf_a, nbuf_a, buf_b, nbuf_b)
                t.start("ring_kv2 fwd: compute")
                blk = m_.attns[s]
                if blk.col.numel() > 0:
                    kv = torch.cat([buf_a, buf_b], dim=1)                      # [2, chunk, H*d]: A of chunk r-s | B of chunk r+s
                    if kv.dtype != kdt:
                        kv = kv.to(kdt)
                    acc_s, l_s, _ = fwd(q, kv[0].view(n, h, d), kv[1].view(n, h, d), blk.crow, ci(blk.col), ei(blk.etype),
                                        gate_c, None, score=m_.score, norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
                    acc += acc_s.to(pdt)
                    l += l_s.to(pdt)
                    del acc_s, l_s, kv
                t.stop("ring_kv2 fwd: compute")
                if s < P - 1 and not m_.no_comm:
                    t.start("ring_kv2 fwd: wait")
                    ring.wait(works)
                    t.stop("ring_kv2 fwd: wait")
                    buf_a, nbuf_a = nbuf_a, buf_a
                    buf_b, nbuf_b = nbuf_b, buf_b
            del buf_a, buf_b, nbuf_a, nbuf_b
        ctx.module = m_
        ctx.save_for_backward(q, k, v, gate_c if gate_c is not None else torch.zeros(0, device=q.device))
        ctx.has_gate = gate is not None
        ctx.gate_dtype = gate.dtype if gate is not None else None
        ctx.pdt = pdt
        return acc, l

    @staticmethod
    def backward(ctx, dacc, dl):
        q, k, v, gate_c = ctx.saved_tensors
        m_ = ctx.module
        ring, P = m_.ring, m_.ring.size
        h, d = m_.h, m_.d
        hd = h * d
        n = q.shape[0]
        h_a = m_.h_a
        _, bwd, ci, ei = _kernels(m_.backend)
        cdt = m_.comm_dtype or q.dtype
        kdt = cdt if m_.backend == "triton" else q.dtype
        pdt = ctx.pdt
        gate_arg = gate_c if ctx.has_gate else None
        t = _timers()
        with torch.no_grad():
            dacc = dacc.contiguous().to(pdt)
            dl = dl.contiguous().to(pdt)
            m0 = torch.zeros_like(dl)
            dq = torch.zeros(n, h, d, dtype=pdt, device=q.device)
            dgate = None
            buf_a, buf_b = _RingAttnKV2._pack_halves(k, v, h_a, cdt)
            nbuf_a, nbuf_b = (torch.empty_like(buf_a), torch.empty_like(buf_b)) if P > 1 else (None, None)
            in_a = in_b = None                                                  # accumulators arriving (A from prev, B from next)
            works_kv = works_d = None
            tot_a = tot_b = None
            for s in range(P):
                if s < P - 1 and not m_.no_comm:
                    works_kv = _RingAttnKV2._send_recv2(ring, buf_a, nbuf_a, buf_b, nbuf_b)
                t.start("ring_kv2 bwd: compute")
                blk = m_.attns[s]
                if blk.col.numel() > 0:
                    kv = torch.cat([buf_a, buf_b], dim=1)
                    if kv.dtype != kdt:
                        kv = kv.to(kdt)
                    dq_s, dk_s, dv_s, dgate_s, _ = bwd(q, kv[0].view(n, h, d), kv[1].view(n, h, d), blk.crow, ci(blk.col),
                                                       ei(blk.etype), gate_arg, None, m0, dacc, dl, score=m_.score,
                                                       norm=m_.norm, clamp=m_.clamp, scale=m_.scale)
                    dq += dq_s.to(pdt)
                    if dgate_s is not None:
                        dgate = dgate_s if dgate is None else dgate + dgate_s
                    dk_s, dv_s = dk_s.reshape(n, hd), dv_s.reshape(n, hd)
                    d_a = torch.stack([dk_s[:h_a], dv_s[:h_a]]).to(pdt)         # contribution to chunk (r-s)'s half A
                    d_b = torch.stack([dk_s[h_a:], dv_s[h_a:]]).to(pdt)         # contribution to chunk (r+s)'s half B
                    del dq_s, dk_s, dv_s, kv
                else:
                    d_a = torch.zeros(2, h_a, hd, dtype=pdt, device=q.device)
                    d_b = torch.zeros(2, n - h_a, hd, dtype=pdt, device=q.device)
                t.stop("ring_kv2 bwd: compute")
                if P == 1 or m_.no_comm:
                    tot_a = d_a if tot_a is None else tot_a + d_a
                    tot_b = d_b if tot_b is None else tot_b + d_b
                    continue
                if s > 0:
                    t.start("ring_kv2 bwd: wait dkv")
                    ring.wait(works_d)
                    t.stop("ring_kv2 bwd: wait dkv")
                    d_a += in_a.to(pdt)
                    d_b += in_b.to(pdt)
                send_a, send_b = d_a.to(cdt).contiguous(), d_b.to(cdt).contiguous()
                in_a, in_b = torch.empty_like(send_a), torch.empty_like(send_b)
                works_d = _RingAttnKV2._send_recv2(ring, send_a, in_a, send_b, in_b)
                del d_a, d_b
                if s < P - 1:
                    t.start("ring_kv2 bwd: wait kv")
                    ring.wait(works_kv)
                    t.stop("ring_kv2 bwd: wait kv")
                    buf_a, nbuf_a = nbuf_a, buf_a
                    buf_b, nbuf_b = nbuf_b, buf_b
            if P > 1 and not m_.no_comm:
                t.start("ring_kv2 bwd: wait dkv")
                ring.wait(works_d)                                              # my own chunk's two accumulators
                t.stop("ring_kv2 bwd: wait dkv")
                tot_a, tot_b = in_a.to(pdt), in_b.to(pdt)
            del buf_a, buf_b, nbuf_a, nbuf_b
            dk = torch.cat([tot_a[0], tot_b[0]]).view(n, h, d)
            dv = torch.cat([tot_a[1], tot_b[1]]).view(n, h, d)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dgate.to(ctx.gate_dtype) if dgate is not None else None, None)


class DistRingAttentionKV(torch.nn.Module):
    """Ring attention with K/V rotation on a (P, 1) grid (module docstring).  q, k, v are the owned chunk with all
    heads [chunk, H, d] (projected by the owner); o is the owned chunk.  `blocks` are the P sub-blocks A[r, c] of
    the local block, one per source chunk c (`split_block_by_source_chunk`), as AttentionBlocks or
    (edge_index, edge_type) tuples; `from_block` splits a whole local block.  comm_dtype (e.g. bfloat16) is the wire
    dtype of the K/V hops and of the dK/dV accumulator; no_comm replaces every hop by nothing (wrong result; the
    kernels then see the owned chunk at every step: compute-only timing)."""

    def __init__(self, grid, blocks, num_heads, head_dim, backend="triton", score="gated", norm="clampexp",
                 clamp=(-5.0, 5.0), eps=1e-6, comm_dtype=None, no_comm=False, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.h_a = grid.chunk // 2                                          # rows [0, h_a) travel clockwise (bidir)
        assert grid.p_cols == 1, "ring attention is a 1D row partition: use a (P, 1) grid"
        assert norm == "clampexp", "the sub-block partials add up without rescaling only for clamp-exp"
        self.grid = grid
        self.h, self.d = num_heads, head_dim
        self.backend, self.score, self.norm, self.clamp, self.eps = backend, score, norm, clamp, eps
        self.scale = 1.0 / (head_dim ** 0.5)
        self.comm_dtype, self.no_comm = comm_dtype, no_comm
        self.ring = RingComm(grid.col_group)                                # all P ranks, in rank order
        self.attns = [b if isinstance(b, AttentionBlock) else AttentionBlock(grid, *b) for b in blocks]
        assert len(self.attns) == self.ring.size == grid.p_rows, (len(self.attns), self.ring.size, grid.p_rows)

    @classmethod
    def from_block(cls, grid, block_edge_index, block_edge_type, num_heads, head_dim, bidir=False, **kw):
        """Split a whole local block: per source chunk (unidirectional: attns[c]) or per ring step (bidir: attns[s])."""
        if bidir:
            rank = dist.get_rank(grid.col_group)
            parts = split_block_bidir(block_edge_index, grid.p_rows, grid.chunk, rank, grid.chunk // 2, block_edge_type)
            return cls(grid, parts, num_heads, head_dim, bidir=True, **kw)
        parts = split_block_by_source_chunk(block_edge_index, grid.p_rows, grid.chunk, block_edge_type, None)
        return cls(grid, [(ei, et) for ei, et, _ in parts], num_heads, head_dim, **kw)

    @torch.compiler.disable
    def forward(self, q_loc, k_loc, v_loc, gate=None):
        fn = _RingAttnKV2 if self.bidir else _RingAttnKV
        acc, l = fn.apply(q_loc.contiguous(), k_loc.contiguous(), v_loc.contiguous(), gate, self)
        return (acc / (l + self.eps).unsqueeze(-1)).to(q_loc.dtype)
