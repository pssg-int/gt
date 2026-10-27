"""Sparsity-aware row collectives: all-gather / reduce-scatter restricted to the rows a rank actually needs.

The graph is static, so which rows of a gathered block a rank touches is known once and for all:
  * column-block gather (K / V side, fused 2D and streamed layers, GP-AG = (P, 1)): rank (r, c) needs the sources that
    have an edge into row block r, i.e. the non-empty COLUMNS of its tile A[r, c];
  * row-block gather (Q side) and the reduce-scatter of the (acc, l) partials: the non-empty ROWS of A[r, c] (an empty
    row has a zero partial, so nothing has to be sent back for it either);
  * the head grid's slice gathers are the same (P_d, P_o) block grid with P_i x smaller chunks.
A SparseRowPlan replaces NCCL's ring all-gather of [G * rows_per_rank, F] by an all-to-all-v that moves only those rows:
the sender compacts its chunk with a static index list (one index_select), the receiver gets the rows it needs in
block-local order (compact index = rank of the row in its sorted need set), and the CSR of the tile is remapped once
to the compact index space.  The reduce-scatter is the transpose: an all-to-all-v of the compact rows back to their
owners plus one index_add_.  Both directions share the same index sets, which are exchanged once inside the group
when the plan is built (zero runtime overhead beyond the index_select / index_add_).

Cost model (RESULTS.md 2026-10-10 / 2026-10-18, Perlmutter): all-to-all moves (G-1)/G of its bytes over the NICs
directly, so per rank it runs at ~0.9x the all-gather rate when the group has one rank per node (1 x 16: a2a 14.8 vs
AG 16.7 GB/s), but at 0.4x / 0.17x when the group spans whole nodes (2 x 16: 13.6 vs 34.3; 4 x 16: 11.6 vs 69.6),
where NCCL's ring all-gather runs 3 of every 4 hops over NVLink.  A plan is therefore only used when the share of the
rows a rank needs is below (a2a rate / AG rate) x margin for the group's node layout (`a2a_over_ag`), unless a
threshold is forced.  Under the random (partition-free) order the share is 1 - exp(-d_attn / P_d) and the same on every
rank (RESULTS 2026-10-18): products 59 % / 20 % / 5 %, papers100M 43 % / 13 % / 3 % at P_d = 64 / 256 / 1024.
"""
import os
import socket

import torch
import torch.distributed as dist

from gt3d.dist.grid2d import _pre_collective, _timers, _gname, _PENDING, wait_gather  # noqa: F401  (wait_gather re-exported)

# a2a / all-gather per-rank rate by (ranks of the group per node, nodes > 1); measured curves at 64-160 MB
# (bench/hw_constants_nccl_64bb.json): 1 x k nodes 0.89, 2 x k 0.40, 4 x k 0.17; inside one node 2 ranks 1.0, 4 ranks 0.92
A2A_OVER_AG = {(1, True): 0.89, (2, True): 0.40, (4, True): 0.17, (1, False): 1.0, (2, False): 1.0, (4, False): 0.92}
MARGIN = float(os.environ.get("GT3D_SPARSE_MARGIN", "0.9"))      # the index_select / index_add_ and the a2a-v's uneven splits


def group_node_layout(group):
    """(ranks of the group on my node, group spans > 1 node); one all_gather_object of the hostnames (plan build time)."""
    g = dist.get_world_size(group)
    if g == 1:
        return 1, False
    names = [None] * g
    dist.all_gather_object(names, socket.gethostname(), group=group)
    mine = names[dist.get_rank(group)]
    return sum(1 for n in names if n == mine), len(set(names)) > 1


def a2a_over_ag(group):
    per_node, multi = group_node_layout(group)
    key = (min(per_node, 4), multi)
    return A2A_OVER_AG.get(key, A2A_OVER_AG[(4, multi)])


def _idx_dtype(n):
    return torch.int32 if n < 2 ** 31 else torch.long


class SparseRowPlan:
    """Rows I need (gather) / hold (reduce-scatter) of a block gathered over `group` in group-rank order, every rank
    contributing `rows_per_rank` rows.  need_local: sorted unique indices into [0, G * rows_per_rank) (device tensor).
    After construction:
      n_need           rows of the compact block (= len(need_local)), its row k is block-local row need_local[k]
      remap            LUT [G * rows_per_rank] block-local -> compact index (-1 when not needed), int32 / int64
      gather(x)        [rows_per_rank, F] -> [n_need, F]            (autograd pair: sparse reduce-scatter)
      reduce_scatter(y)[n_need, F] -> [rows_per_rank, F] (sum over the group's holders of every row)
      share            n_need / (G * rows_per_rank), max over the group (the byte ratio of the sparse collective)."""

    def __init__(self, group, rows_per_rank: int, need_local: torch.Tensor, name: str = "?"):
        self.group, self.rows_per_rank, self.name = group, int(rows_per_rank), name
        g, me = dist.get_world_size(group), dist.get_rank(group)
        self.g, self.me = g, me
        dev = need_local.device
        need_local = need_local.long()
        assert need_local.numel() == 0 or (need_local.min() >= 0 and need_local.max() < g * rows_per_rank), name
        n_blk = g * rows_per_rank
        self.n_need = int(need_local.numel())
        owner = need_local // rows_per_rank
        recv = torch.bincount(owner, minlength=g).cpu()                        # rows I need from every rank
        if g == 1:
            send = recv.clone()
            send_idx = need_local
        else:
            # how many rows every rank needs from me: one all-to-all of the counts ...
            send = torch.empty(g, dtype=torch.long, device=dev)
            dist.all_to_all_single(send, recv.to(dev), group=group)
            send = send.cpu()
            # ... then the index lists themselves (local indices inside my chunk), ordered by receiver
            send_idx = torch.empty(max(int(send.sum()), 1), dtype=torch.long, device=dev)
            dist.all_to_all_single(send_idx[:int(send.sum())], (need_local - owner * rows_per_rank).contiguous(),
                                   output_split_sizes=send.tolist(), input_split_sizes=recv.tolist(), group=group)
            send_idx = send_idx[:int(send.sum())]
        self.recv_counts, self.send_counts = recv.tolist(), send.tolist()
        self.n_send = int(sum(self.send_counts))
        self.send_idx = send_idx.to(_idx_dtype(rows_per_rank)).contiguous()
        # reduce-scatter accumulation without atomics: the received rows permuted into send_idx order form contiguous
        # segments per owned row (segment_csr); perm / indptr are static
        self.perm = torch.argsort(send_idx, stable=True).to(torch.int32).contiguous()
        self.indptr = torch.zeros(rows_per_rank + 1, dtype=torch.long, device=dev)
        self.indptr[1:] = torch.cumsum(torch.bincount(send_idx, minlength=rows_per_rank), 0)
        self.indptr = self.indptr.to(torch.int32).contiguous()
        self._sum_mats = {}                                                          # dtype -> ones-valued CSR [rows_per_rank, n_send]
        remap = torch.full((n_blk,), -1, dtype=_idx_dtype(n_blk), device=dev)
        remap[need_local] = torch.arange(self.n_need, dtype=remap.dtype, device=dev)
        self.remap = remap
        self.need_local = need_local.to(_idx_dtype(n_blk))
        # byte ratio of the sparse collective vs the dense one, max over the group (what the slowest rank moves)
        sh = torch.tensor([self.n_need / n_blk, self.n_send / n_blk], dtype=torch.float64, device=dev)
        if g > 1:
            dist.all_reduce(sh, op=dist.ReduceOp.MAX, group=group)
        self.share, self.share_send = float(sh[0]), float(sh[1])
        self.ratio = a2a_over_ag(group)                                            # primitive penalty of the layout

    def worthwhile(self, threshold=None):
        """Use the sparse collective?  threshold: forced share limit (1.0 = always, 0 = never); None = ratio x margin."""
        lim = self.ratio * MARGIN if threshold is None else threshold
        return max(self.share, self.share_send) <= lim

    def drop_remap(self):
        """Free the block-local -> compact LUT once the tiles have been remapped (only need_local is used at run time)."""
        self.remap = None

    def nbytes(self, seen=None):
        seen = set() if seen is None else seen
        total = 0
        for t in [self.send_idx, self.remap, self.need_local, self.perm, self.indptr] + [m.values() for m in self._sum_mats.values()]:
            if t is not None and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); total += t.numel() * t.element_size()
        return total

    # ---- raw collectives --------------------------------------------------------------------
    @staticmethod
    def _aligned(x):
        """[n, F] -> ([n, F_pad], F): rows padded to a multiple of 16 bytes.  With per-row offsets that are not 16-byte
        aligned (e.g. the [acc | l] payload of width H d + H = 132 / 260 in bf16) NCCL's send / recv kernels and the
        index_select run ~2x slower (profile 2026-10-25: dim 132 a2a-v 15.3 ms vs dim 136 7.2 ms per 1.7 GB); one
        extra pass over the payload is far cheaper."""
        F = x.shape[1]
        unit = 16 // x.element_size()
        pad = (-F) % unit
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x, F

    def _send_buffer(self, x):
        return x.index_select(0, self.send_idx)                                    # [n_send, F]

    def gather(self, x, async_op=False):
        """x [rows_per_rank, ...] -> [n_need, ...]; async_op -> (work, finalize)."""
        shp = x.shape[1:]
        x = x.contiguous().view(x.shape[0], -1)
        if self.g == 1:
            out = x.index_select(0, self.send_idx).view(-1, *shp)
            return (None, (lambda: out)) if async_op else out
        x, F = self._aligned(x)
        send = self._send_buffer(x)
        out = x.new_empty(max(self.n_need, 1), x.shape[1])
        if not async_op:
            _pre_collective(self.group)
        work = dist.all_to_all_single(out[:self.n_need], send, output_split_sizes=self.recv_counts,
                                      input_split_sizes=self.send_counts, group=self.group, async_op=async_op)
        fin = lambda: out[:self.n_need, :F].view(-1, *shp)                        # noqa: E731
        if async_op:
            _KEEP[id(work)] = send                                                 # the send buffer must outlive the op
            return work, fin
        return fin()

    def reduce_scatter(self, y, async_op=False):
        """y [n_need, ...] partial rows -> [rows_per_rank, ...] (sum over the group); async_op -> (work, finalize)."""
        shp = y.shape[1:]
        y = y.contiguous().view(y.shape[0], -1)
        if self.g == 1:
            out = self._accumulate(y).view(-1, *shp)
            return (None, (lambda: out)) if async_op else out
        y, F = self._aligned(y)
        recv = y.new_empty(max(self.n_send, 1), y.shape[1])
        if not async_op:
            _pre_collective(self.group)
        work = dist.all_to_all_single(recv[:self.n_send], y, output_split_sizes=self.send_counts,
                                      input_split_sizes=self.recv_counts, group=self.group, async_op=async_op)

        def fin():
            return self._accumulate(recv[:self.n_send])[:, :F].contiguous().view(-1, *shp)

        if async_op:
            _KEEP[id(work)] = y
            return work, fin
        return fin()

    def _accumulate(self, rows):
        """[n_send, F] received rows -> [rows_per_rank, F] sums over the holders of every row, as a sparse-dense product:
        a static ones-valued CSR S [rows_per_rank, n_send] (row i lists the received rows owned by i) times the rows
        (cuSPARSE SpMM: one coalesced read of the rows, fp32 accumulation inside for 16-bit payloads, deterministic).
        bench/bench_sparse_accumulate.py: 1.5-2.5 ms per 0.3-1.9 GB payload, 1.7-2x faster than permutation +
        segment_csr and 3-10x faster than index_add_ (atomics), and never slower than a plain copy."""
        if self.n_send == 0:
            return torch.zeros(self.rows_per_rank, rows.shape[1], dtype=rows.dtype, device=rows.device)
        S = self._sum_mats.get(rows.dtype)
        if S is None:
            ones = torch.ones(self.n_send, dtype=rows.dtype, device=rows.device)
            S = self._sum_mats[rows.dtype] = torch.sparse_csr_tensor(self.indptr, self.perm, ones, size=(self.rows_per_rank, self.n_send))
        return torch.sparse.mm(S, rows)

    def __repr__(self):
        return (f"SparseRowPlan({self.name}: G={self.g} rows/rank={self.rows_per_rank} need {self.n_need} "
                f"share {self.share:.3f} send {self.share_send:.3f} a2a/AG {self.ratio:.2f})")


_KEEP = {}                                                                          # id(work) -> buffer alive until wait


def wait_work(work):
    if work is not None:
        work.wait()
        _KEEP.pop(id(work), None)


# ---- autograd pairing (mirrors grid2d.all_gather_rows / reduce_scatter_rows) ------------------------
class _SparseAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        return plan.gather(x)

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd reduce-scatter ({_gname(ctx.plan.group)})")
        out = ctx.plan.reduce_scatter(grad)
        t.stop(f"bwd reduce-scatter ({_gname(ctx.plan.group)})")
        return out, None


class _SparseReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, plan):
        ctx.plan = plan
        return plan.reduce_scatter(y)

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd all-gather ({_gname(ctx.plan.group)})")
        out = ctx.plan.gather(grad)
        t.stop(f"bwd all-gather ({_gname(ctx.plan.group)})")
        return out, None


class _SparseAllGatherStart(torch.autograd.Function):
    """Asynchronous sparse gather: returns the (not yet valid) buffer; grid2d.wait_gather makes it usable."""

    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        work, fin = plan.gather(x, async_op=True)
        out = fin()
        if work is not None:
            _PENDING[out.data_ptr()] = work
        return out

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd reduce-scatter ({_gname(ctx.plan.group)})")
        out = ctx.plan.reduce_scatter(grad)
        t.stop(f"bwd reduce-scatter ({_gname(ctx.plan.group)})")
        return out, None


@torch.compiler.disable
def sparse_all_gather(x, plan):
    return _SparseAllGather.apply(x, plan)


@torch.compiler.disable
def sparse_reduce_scatter(y, plan):
    return _SparseReduceScatter.apply(y, plan)


@torch.compiler.disable
def sparse_all_gather_start(x, plan):
    return _SparseAllGatherStart.apply(x, plan)


# ---- plans from a tile's CSR -------------------------------------------------------------------------
def tile_need_sets(crow, col, n_cols):
    """(non-empty rows, referenced columns) of a CSR tile, both sorted int64."""
    deg = crow[1:] - crow[:-1]
    rows = torch.nonzero(deg > 0).squeeze(1).long()
    cols = torch.unique(col.long())
    return rows, cols


def compact_csr(crow, rows_need, col, col_remap):
    """CSR restricted to the rows `rows_need` (which must contain every non-empty row) with the columns remapped:
    crow_c [len(rows_need) + 1], col_c [E].  Rows outside rows_need own no edge, so the edges stay contiguous."""
    crow = crow.long()
    crow_c = torch.cat([crow[rows_need.long()], crow[-1:]])
    col_c = col_remap[col.long()]
    assert bool((col_c >= 0).all()), "a referenced column is not in the column need set"
    return crow_c, col_c.long()
