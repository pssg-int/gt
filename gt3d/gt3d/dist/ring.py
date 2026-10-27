"""Ring attention (1D context parallelism with rotating keys/values) for structure-masked sparse attention.

Layout: a (P, 1) Grid2D, i.e. a pure row (query) partition.  Rank r owns node chunk r and computes the complete
attention rows of its chunk.  Instead of all-gathering the whole K/V (the DAC'26 "GP-AG" layout = grid (P, 1) with
the fused layer), the layer-input chunks travel around a ring of the P ranks: at step s rank r holds chunk
c = (r - s) mod P, projects K, V and the GCN message of that chunk locally and processes the sub-block A[r, c]
(the edges of its rows whose sources lie in chunk c) while the chunk is already on its way to rank r + 1.  The
unnormalised partials (acc, l) and the GCN aggregation of the P sub-blocks add up (clamp-exp needs no rescaling);
after P steps the rank holds its rows' complete attention.  This is Ring Attention (Liu et al. 2023) with the
block-sparse mask of the graph: the "blocks" of the sequence are node chunks, the per-step kernel is the fused
CSR kernel on one sub-block.

Backward (Ring Attention / FlashAttention style): only the owned input is saved.  The chunks travel again
(K, V and X W_g are recomputed per hop); dQ accumulates locally; the input gradient of every chunk travels WITH
the chunk as an accumulator to which each rank adds its contribution; after the last hop it lands on the owner.

Communication per rank and layer (chunk = N/P rows, hidden d): forward (P-1) hops of chunk x d (the input in the
comm dtype; sending x instead of K and V halves Ring Attention's traffic exactly as the fused 2D layer does);
backward (P-1) hops of x + P hops of the dx accumulator.  In bytes this equals the fused (P, 1) layer's
all-gather + reduce-scatter (NCCL's ring algorithms move the same volume), but the transfers are point-to-point
between ring neighbours, they overlap the step's compute, and the per-rank memory is O(N/P) instead of O(N):
no K/V/x of the whole graph is ever materialised.  What the ring cannot shrink: every rank still runs the kernels
over all N rows of K/V per layer (P passes over N/P-row sub-blocks), so the per-row kernel overhead is O(N) per
rank, and the P dependent hops add latency (the backward hop s+1 needs the neighbour's step-s contribution).

P2P safety: every step posts one batched {isend to next, irecv from prev} per travelling buffer, and every rank
issues the same sequence of batched groups, so the NCCL stream orders are consistent around the ring.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

from gt3d.dist.grid2d import Grid2D


def _timers():
    from axonn import axonn as ax
    return ax.get_timers()


class RingComm:
    """Ring over the ranks of `group` in group-rank order: next = +1, prev = -1."""

    def __init__(self, group):
        self.group = group
        self.size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.next = dist.get_global_rank(group, (self.rank + 1) % self.size)
        self.prev = dist.get_global_rank(group, (self.rank - 1) % self.size)

    def send_recv(self, send_buf, recv_buf):
        """Post {isend(send_buf -> next), irecv(recv_buf <- prev)} as ONE NCCL group; returns the work list.
        send_buf must stay untouched and recv_buf unread until wait()."""
        assert send_buf.is_contiguous() and recv_buf.is_contiguous()
        ops = [dist.P2POp(dist.isend, send_buf, self.next, group=self.group),
               dist.P2POp(dist.irecv, recv_buf, self.prev, group=self.group)]
        return dist.batch_isend_irecv(ops)

    @staticmethod
    def wait(works):
        for w in works:
            w.wait()                                      # the current stream waits for the transfer


def split_block_by_source_chunk(edge_index_local, p, chunk, etype=None, weight=None):
    """Split the local block of a (P, 1) grid (src = column-local = global id, dst = row-local) into the P
    sub-blocks A[r, c] by the source chunk c = src // chunk (= the owner rank of the source), remapping src to
    the chunk-local index.  Sub-block c is processed at the ring step where chunk c arrives."""
    src, dst = edge_index_local[0], edge_index_local[1]
    c_of = src // chunk
    out = []
    for c in range(p):
        m = c_of == c
        out.append((torch.stack([src[m] - c * chunk, dst[m]]),
                    etype[m] if etype is not None else None,
                    weight[m] if weight is not None else None))
    return out


class _RingAttnGCN(torch.autograd.Function):
    """Owned rows of the payload (acc | l | gcn partial), [chunk, h*d + h + dim], summed over the P ring steps.
    Saved for backward: the owned input only (plus the weights)."""

    @staticmethod
    def _kernels(layer):
        if layer.backend == "triton":
            from gt3d.attention.triton_kernel import attn_fwd_partial, attn_bwd_partial
            return attn_fwd_partial, attn_bwd_partial, lambda t: t, lambda t: t
        from gt3d.attention.backend_torch import torch_attn_fwd_partial, torch_attn_bwd_partial
        return torch_attn_fwd_partial, torch_attn_bwd_partial, lambda t: t.long(), lambda t: (None if t is None else t.long())

    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, gate, layer):
        L, ring = layer, layer.ring
        P = ring.size
        h, d = L.num_heads, L.d_head
        hd = h * d
        fwd, _, ci, ei = _RingAttnGCN._kernels(L)
        gate_c = gate.to(torch.float32 if L.backend == "triton" else x.dtype)
        cdt = L.comm_dtype or x.dtype
        with_gcn = getattr(L, "with_gcn", True) and wg is not None          # False: attention only (benchmarks)
        no_comm = getattr(L, "no_comm", False)                              # True: benchmarks only (no hops, wrong result)
        t = _timers()
        with torch.no_grad():
            n, dim = x.shape
            width = hd + h + (dim if with_gcn else 0)                       # payload columns: acc | l | [gcn partial]
            q = (x @ wq.t()).view(n, h, d)                                  # owned rows: no gather
            buf = x.detach().to(cdt, copy=True).contiguous()               # the chunk currently held (my own at step 0)
            nbuf = torch.empty_like(buf) if P > 1 else None
            packed = None
            works = None
            for s in range(P):
                c = (ring.rank - s) % P                                     # source chunk held at this step
                if s < P - 1 and not no_comm:
                    works = ring.send_recv(buf, nbuf)                       # hop s: overlaps this step's compute
                t.start("ring fwd: compute")
                xs = buf.to(x.dtype)
                blk = L.attns[c]
                if packed is None:
                    packed = torch.zeros(n, width, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64,
                                         device=x.device)
                if with_gcn and L.gcn_blocks[c].col.numel() > 0:
                    a, _ = L.gcn_blocks[c].matrices(x.dtype)
                    packed[:, hd + h:] += torch.sparse.mm(a, xs @ wg.t()).to(packed.dtype)
                if blk.col.numel() > 0:
                    ks = (xs @ wk.t()).view(-1, h, d)
                    vs = (xs @ wv.t()).view(-1, h, d)
                    acc_s, l_s, _ = fwd(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None,
                                        score=L.attn_score, norm="clampexp", clamp=L.attn_clamp, scale=L.attn_scale)
                    packed[:, :hd] += acc_s.view(n, hd).to(packed.dtype)
                    packed[:, hd:hd + h] += l_s.to(packed.dtype)
                    del ks, vs, acc_s, l_s
                t.stop("ring fwd: compute")
                if s < P - 1 and not no_comm:
                    t.start("ring fwd: wait")
                    ring.wait(works)
                    t.stop("ring fwd: wait")
                    buf, nbuf = nbuf, buf                                   # received chunk becomes current; old one is free
        ctx.save_for_backward(x, wq, wk, wv, wg if with_gcn else torch.zeros(0, device=x.device), gate)
        ctx.layer = L
        ctx.with_gcn = with_gcn
        ctx.gate_c_dtype = gate_c.dtype
        ctx.packed_dtype = packed.dtype
        return packed

    @staticmethod
    def backward(ctx, dpo):
        x, wq, wk, wv, wg, gate = ctx.saved_tensors
        L, ring = ctx.layer, ctx.layer.ring
        P = ring.size
        h, d = L.num_heads, L.d_head
        hd = h * d
        _, bwd, ci, ei = _RingAttnGCN._kernels(L)
        gate_c = gate.to(ctx.gate_c_dtype)
        cdt = L.comm_dtype or x.dtype
        pdt = ctx.packed_dtype
        with_gcn = ctx.with_gcn
        no_comm = getattr(L, "no_comm", False)
        t = _timers()
        with torch.no_grad():
            n, dim = x.shape
            dpo = dpo.contiguous()
            dacc = dpo[:, :hd].contiguous().to(pdt).view(n, h, d)
            dl = dpo[:, hd:hd + h].contiguous().to(pdt)
            dagg = dpo[:, hd + h:].contiguous().to(x.dtype) if with_gcn else None
            q = (x @ wq.t()).view(n, h, d)
            m0 = torch.zeros_like(dl)                                       # clamp-exp partials have m = 0
            dq = torch.zeros_like(q)
            dwk, dwv = torch.zeros_like(wk), torch.zeros_like(wv)
            dwg = torch.zeros_like(wg) if with_gcn else None
            dgate = None
            buf = x.detach().to(cdt, copy=True).contiguous()
            nbuf = torch.empty_like(buf) if P > 1 else None
            dx_in = torch.empty_like(buf) if P > 1 else None               # accumulator arriving from prev
            works_x = works_dx = None
            dx_total = None
            for s in range(P):
                c = (ring.rank - s) % P
                if s < P - 1 and not no_comm:
                    works_x = ring.send_recv(buf, nbuf)                     # x for step s+1
                t.start("ring bwd: compute")
                xs = buf.to(x.dtype)
                blk = L.attns[c]
                dxs = torch.zeros(n, dim, dtype=x.dtype, device=x.device)   # contribution to chunk c's input gradient
                if with_gcn and L.gcn_blocks[c].col.numel() > 0:
                    _, at = L.gcn_blocks[c].matrices(x.dtype)
                    dp_s = torch.sparse.mm(at, dagg)                        # grad of the GCN projection xs @ wg^T
                    dwg += dp_s.t() @ xs
                    dxs += dp_s @ wg
                    del dp_s
                if blk.col.numel() > 0:
                    ks = (xs @ wk.t()).view(-1, h, d)
                    vs = (xs @ wv.t()).view(-1, h, d)
                    dq_s, dks, dvs, dgate_s, _ = bwd(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None,
                                                     m0, dacc, dl, score=L.attn_score, norm="clampexp",
                                                     clamp=L.attn_clamp, scale=L.attn_scale)
                    del ks, vs
                    dq += dq_s.to(q.dtype)
                    if dgate_s is not None:
                        dgate = dgate_s if dgate is None else dgate + dgate_s
                    dks, dvs = dks.to(x.dtype).view(-1, hd), dvs.to(x.dtype).view(-1, hd)
                    dwk += dks.t() @ xs
                    dwv += dvs.t() @ xs
                    dxs += dks @ wk
                    dxs += dvs @ wv
                    del dq_s, dks, dvs
                t.stop("ring bwd: compute")
                if P == 1 or no_comm:
                    dx_total = dxs if dx_total is None else dx_total + dxs
                    continue
                if s > 0:
                    t.start("ring bwd: wait dx")
                    ring.wait(works_dx)                                     # chunk c's accumulator from prev (its step s-1)
                    t.stop("ring bwd: wait dx")
                    dxs += dx_in.to(x.dtype)
                dx_send = dxs.to(cdt).contiguous()                          # accumulate in the comm dtype (as the reduce-scatter does)
                dx_in = torch.empty_like(dx_send)
                works_dx = ring.send_recv(dx_send, dx_in)                   # hop s of the accumulator (P hops in total)
                del dxs
                if s < P - 1:
                    t.start("ring bwd: wait x")
                    ring.wait(works_x)
                    t.stop("ring bwd: wait x")
                    buf, nbuf = nbuf, buf
            if P > 1 and not no_comm:
                t.start("ring bwd: wait dx")
                ring.wait(works_dx)                                         # my own chunk's accumulator, complete
                t.stop("ring bwd: wait dx")
                dx = dx_in.to(x.dtype)
            else:
                dx = dx_total
            del buf, nbuf
            dq2 = dq.view(n, hd)                                            # Q side: owned rows, no communication
            dwq = dq2.t() @ x
            dx = dx + dq2 @ wq
        return dx, dwq, dwk, dwv, dwg, (dgate.to(gate.dtype) if dgate is not None else None), None


from gt3d.models.exphormer_dist import DistExphormerLayerStreamed  # noqa: E402  (after the Function: no cycle at import time)


class DistExphormerLayerRing(DistExphormerLayerStreamed):
    """Exphormer layer on a (P, 1) grid with ring attention (module docstring).  Parameters and the owned-row
    tail (normalisation of the partials, GCN bias, dropout, norms, FFN) are those of the streamed layer."""

    self_checkpointing = True

    def __init__(self, grid: Grid2D, gcn_blocks, attn_blocks, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                 comm_dtype=None, tail_checkpoint=False, with_gcn=True):
        assert grid.p_cols == 1, "ring attention is a 1D row partition: use a (P, 1) grid"
        # with_gcn=False: attention-only layer (gcn_blocks is a list of None; no GCN parameters, see _tail)
        super().__init__(grid, gcn_blocks, attn_blocks, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                         comm_dtype=comm_dtype, rs_dtype=None, tail_checkpoint=tail_checkpoint, with_gcn=with_gcn)
        self.ring = RingComm(grid.col_group)                                # all P ranks, in rank order
        assert self.ring.size == grid.p_rows == len(gcn_blocks) == len(attn_blocks), \
            (self.ring.size, grid.p_rows, len(gcn_blocks), len(attn_blocks))

    def forward(self, x, edge_type_emb):
        h, d = self.num_heads, self.d_head
        gate = self.e(edge_type_emb).view(-1, h, d)
        wg = self.gcn.lin.weight if self.with_gcn else None
        packed_o = _RingAttnGCN.apply(x, self.q.weight, self.k.weight, self.v.weight, wg, gate, self)
        if self.tail_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._tail, x, packed_o, use_reentrant=False)
        return self._tail(x, packed_o)
