"""2D (rows x columns) process grid for distributed masked sparse attention, v0 (no head axis).
Process groups are AxoNN's intra-layer groups (see Grid2D.__init__); AxoNN and Plexus are used unmodified.

Layout
------
P = P_d * P_o ranks are arranged as rank = r * P_o + c with r the row-block index (destination /
query side) and c the column-block index (source / key-value side).

Nodes are padded to N_pad = P * chunk and split into P contiguous chunks of `chunk` nodes.
Rank (r, c) OWNS chunk index  c * P_d + r,  i.e. node ids [ (c*P_d + r)*chunk, +chunk ).
Consequences:
  * column block c = union of the chunks owned by ranks (., c)  = ids [c*P_d*chunk, (c+1)*P_d*chunk)
    -> contiguous, and its owners are exactly the column group {(r, c) : r}.
  * row block r = union of the chunks owned by ranks (r, .)  (non-contiguous, P_o chunks)
    -> its owners are exactly the row group {(r, c) : c}.
Rank (r, c) computes the attention block A[row block r, column block c].  Communication:
  * K, V for column block c : all-gather inside the column group   (owners == consumers)
  * Q      for row block r  : all-gather inside the row group
  * partial outputs for row block r (one per column block) : reduce-scatter inside the row group,
    landing exactly on the owner of every row.
Every dense op (projections, FFN, norms, loss) works on the owned chunk [chunk, d] with no
communication ("node-sharded" / sequence-parallel layout).
"""
import os

import torch
import torch.distributed as dist

# process group -> short name ("row" / "col"), filled by Grid2D / Grid3D for the timers of the backward collectives
GROUP_NAMES = {}


def _timers():
    """AxoNN's CUDA-event timers (a no-op object when AxoNN is not initialised, e.g. Grid3D / gloo tests)."""
    try:
        from axonn import axonn as ax
        t = ax.get_timers()
    except Exception:
        t = None
    return t if t is not None else _NoTimers()


class _NoTimers:
    def start(self, *a):
        pass

    def stop(self, *a):
        pass


def _gname(group):
    return GROUP_NAMES.get(id(group), "?")


# GT3D_SYNC_COLLECTIVES (default 1): host-synchronise the current stream before every SYNCHRONOUS collective, so that
# the previous collective — possibly on another communicator — has completed before this one is posted to NCCL and
# the network.  Without it, consecutive collectives on two different cross-node communicators (a row group and a
# column group, or a cross-node head group) are in flight together on the same NIC and run 1.5-3x slower than their
# alpha-beta cost (RESULTS 2026-10-17, "two cross-node groups": (2,2,4) 188 -> 120 ms, (2,1,8) 203 -> 117, (2,4,2)
# 230 -> 144, fused 2x8 241 -> 181 ms per layer at 16 GPUs; single-cross-node-axis shapes unchanged).  =2 adds a
# barrier over the group; =0 disables.  Asynchronous gathers (the streamed layer's prefetch) are never synchronised.
SYNC_COLLECTIVES = int(os.environ.get("GT3D_SYNC_COLLECTIVES", "1"))


def _pre_collective(group):
    if SYNC_COLLECTIVES:
        torch.cuda.current_stream().synchronize()
        if SYNC_COLLECTIVES >= 2 and dist.get_world_size(group) > 1:
            dist.barrier(group=group)


class Grid2D:
    def __init__(self, p_rows: int, p_cols: int, num_nodes: int, n_pad: int = None, layout: str = "row_major",
                 timers: bool = False, use_axonn: bool = None):
        """layout: 'row_major' -> rank = r*P_o + c (row groups are consecutive ranks, i.e. intra-node for
        P_o <= GPUs per node); 'col_major' -> rank = c*P_d + r (column groups consecutive: the K/V-side
        input gather and dX reduce-scatter stay on NVLink instead).

        Process groups come from AxoNN (axonn.init / comm_handle.get_intra_layer_groups): AxoNN arranges the
        intra-layer ranks as [G_d, G_r, G_c] with the column rank fastest, its "inner" group = ranks that share
        the row rank (consecutive) and its "outer" group = ranks that share the column rank (stride G_c).
          row_major: (G_r, G_c) = (P_d, P_o); r = AxoNN row rank, c = AxoNN column rank;
                     row group = inner group, column group = outer group.
          col_major: (G_r, G_c) = (P_o, P_d); r = AxoNN column rank, c = AxoNN row rank;
                     row group = outer group, column group = inner group.
        Both reproduce the rank formulas of the original Grid2D exactly.  `timers` enables AxoNN's CUDA-event
        timers (ax.get_timers()).  use_axonn=False (or a non-NCCL backend, e.g. CPU/gloo tests) keeps the
        original torch.distributed.new_group construction."""
        assert dist.is_initialized()
        world = dist.get_world_size()
        assert world == p_rows * p_cols, (world, p_rows, p_cols)
        self.p_rows, self.p_cols, self.layout = p_rows, p_cols, layout
        self.rank = dist.get_rank()
        if layout == "row_major":
            self.r, self.c = divmod(self.rank, p_cols)
        elif layout == "col_major":
            self.c, self.r = divmod(self.rank, p_rows)
        else:
            raise ValueError(layout)
        self.num_nodes = num_nodes
        if n_pad is None:
            self.chunk = -(-num_nodes // world)
        else:
            assert n_pad % world == 0 and n_pad >= num_nodes, (n_pad, world, num_nodes)
            self.chunk = n_pad // world
        self.n_pad = self.chunk * world
        rank_of = (lambda r, c: r * p_cols + c) if layout == "row_major" else (lambda r, c: c * p_rows + r)
        self.rank_of = rank_of
        if use_axonn is None:
            use_axonn = dist.get_backend() == "nccl"
        self.use_axonn = use_axonn
        if use_axonn:
            from axonn import axonn as ax
            g_r, g_c = (p_rows, p_cols) if layout == "row_major" else (p_cols, p_rows)
            if not ax.is_initialized:
                ax.init(G_inter=1, G_data=1, G_intra_r=g_r, G_intra_c=g_c, G_intra_d=1, enable_internal_timers=timers)
            elif timers:
                ax.enable_timers = True
            h = ax.comm_handle
            ax_col = self.rank % g_c
            ax_row = (self.rank // g_c) % g_r
            assert (ax_row, ax_col) == ((self.r, self.c) if layout == "row_major" else (self.c, self.r))
            inner, outer, depth = h.get_intra_layer_groups((g_r, g_c, 1))   # cached for the init dims
            self.row_group, self.col_group = (inner, outer) if layout == "row_major" else (outer, inner)
            self.depth_group = depth
            self.intra_group = h.intra_layer_group                           # all P ranks
        else:
            # torch.distributed requires every rank to create every group
            self.row_groups = [dist.new_group([rank_of(r, c) for c in range(p_cols)]) for r in range(p_rows)]
            self.col_groups = [dist.new_group([rank_of(r, c) for r in range(p_rows)]) for c in range(p_cols)]
            self.row_group = self.row_groups[self.r]
            self.col_group = self.col_groups[self.c]
            self.depth_group = None
            self.intra_group = None
        GROUP_NAMES[id(self.row_group)] = "row"
        GROUP_NAMES[id(self.col_group)] = "col"
        lo, hi = self.owned_range()
        self.valid_mask = torch.arange(lo, hi) < num_nodes            # owned rows that are real nodes (CPU)

    def to(self, device):
        self.valid_mask = self.valid_mask.to(device)
        return self

    def take_owned(self, x_global):
        """Owned chunk of a global per-node tensor, zero-padded beyond num_nodes."""
        lo, hi = self.owned_range()
        out = x_global.new_zeros((self.chunk,) + tuple(x_global.shape[1:]))
        n = min(hi, self.num_nodes) - lo
        if n > 0:
            out[:n] = x_global[lo:lo + n]
        return out

    # ---- ownership ----------------------------------------------------------------------
    def chunk_of(self, r, c):
        return c * self.p_rows + r

    def owned_range(self, r=None, c=None):
        r = self.r if r is None else r
        c = self.c if c is None else c
        k = self.chunk_of(r, c)
        return k * self.chunk, (k + 1) * self.chunk

    def owner_of(self, node_ids):
        """(r, c) of the owner of each (padded) node id."""
        k = node_ids // self.chunk
        return k % self.p_rows, k // self.p_rows

    def col_block_range(self, c=None):
        c = self.c if c is None else c
        return c * self.p_rows * self.chunk, (c + 1) * self.p_rows * self.chunk

    def row_block_ids(self, r=None):
        """Global ids of row block r in the order produced by all-gather over the row group."""
        r = self.r if r is None else r
        return torch.cat([torch.arange(*self.owned_range(r, c)) for c in range(self.p_cols)])

    def row_block_ranges(self, r=None):
        """(lo, hi) global id ranges of the chunks of row block r, in row-local (all-gather) order."""
        r = self.r if r is None else r
        return [self.owned_range(r, c) for c in range(self.p_cols)]

    @property
    def n_row_block(self):
        return self.p_cols * self.chunk

    def global_to_row_local(self, gid):
        """Global id -> local index inside row block r (all-gather order: chunk of (r,0), (r,1), ...)."""
        r, c = self.owner_of(gid)
        return c * self.chunk + (gid - (self.chunk_of(r, c) * self.chunk))

    def global_to_col_local(self, gid):
        """Global id -> local index inside column block c (all-gather order: chunk of (0,c), (1,c), ...)."""
        lo, _ = self.col_block_range(self.owner_of(gid)[1])
        return gid - lo

    # ---- local block of the attention graph ------------------------------------------
    def local_block(self, edge_index, edge_type=None):
        """Edges (src -> dst) with dst in my row block and src in my column block, remapped to
        local (row-local dst, col-local src) indices.  Used for tests / small graphs; real
        data comes pre-partitioned."""
        src, dst = edge_index[0], edge_index[1]
        r_dst, _ = self.owner_of(dst)
        _, c_src = self.owner_of(src)
        m = (r_dst == self.r) & (c_src == self.c)
        ei = torch.stack([self.global_to_col_local(src[m]), self.global_to_row_local(dst[m])])
        return ei, (edge_type[m] if edge_type is not None else None)


# ---- collectives with autograd ---------------------------------------------------------
# The raw collectives are Plexus's row collectives (plexus.gcn_conv_rs._all_gather_rows / _reduce_scatter_rows:
# 2-D [rows, cols] tensors, optional wire dtype, optional async); gt3d adds the autograd pairing
# (all-gather <-> reduce-scatter) and the [rows, ...] -> [rows, cols] flattening.  Plexus is used unmodified.
from plexus.gcn_conv_rs import _all_gather_rows as _plx_all_gather_rows, _reduce_scatter_rows as _plx_reduce_scatter_rows


def _flat(x):
    return x.contiguous().view(x.shape[0], -1)


def raw_all_gather_rows(x, group, async_op=False):
    """[chunk, ...] -> [G*chunk, ...] (group rank order); async_op -> (work, finalize)."""
    if async_op and dist.get_world_size(group) == 1:           # Plexus asserts not async_op for a 1-rank group
        return None, (lambda: x)
    if not async_op:
        _pre_collective(group)
    out = _plx_all_gather_rows(_flat(x), group, wire_dtype=None, async_op=async_op)
    if async_op:
        work, fin = out
        return work, (lambda: fin().view(-1, *x.shape[1:]))
    return out.view(-1, *x.shape[1:])


def raw_reduce_scatter_rows(x, group):
    """[G*chunk, ...] partial sums -> [chunk, ...] (my rank's rows)."""
    _pre_collective(group)
    return _plx_reduce_scatter_rows(_flat(x), group, wire_dtype=None).view(-1, *x.shape[1:])


class _AllGatherRows(torch.autograd.Function):
    """forward: all-gather [chunk, ...] -> [G*chunk, ...] over `group`; backward: reduce-scatter."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return raw_all_gather_rows(x, group)

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd reduce-scatter ({_gname(ctx.group)})")
        out = raw_reduce_scatter_rows(grad, ctx.group)
        t.stop(f"bwd reduce-scatter ({_gname(ctx.group)})")
        return out, None


class _ReduceScatterRows(torch.autograd.Function):
    """forward: reduce-scatter [G*chunk, ...] -> [chunk, ...] (sum) over `group`; backward: all-gather."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return raw_reduce_scatter_rows(x, group)

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd all-gather ({_gname(ctx.group)})")
        out = raw_all_gather_rows(grad, ctx.group)
        t.stop(f"bwd all-gather ({_gname(ctx.group)})")
        return out, None


@torch.compiler.disable
def all_gather_rows(x, group):
    return _AllGatherRows.apply(x, group)


@torch.compiler.disable
def reduce_scatter_rows(x, group):
    return _ReduceScatterRows.apply(x, group)


# ---- asynchronous all-gather: start() returns the (not yet valid) buffer, wait() makes it usable ----
_PENDING = {}


class _AllGatherRowsStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        work, fin = raw_all_gather_rows(x, group, async_op=True)
        out = fin()                                      # the buffer (valid after work.wait())
        if work is not None:
            _PENDING[out.data_ptr()] = work
        return out

    @staticmethod
    def backward(ctx, grad):
        t = _timers(); t.start(f"bwd reduce-scatter ({_gname(ctx.group)})")
        out = raw_reduce_scatter_rows(grad, ctx.group)
        t.stop(f"bwd reduce-scatter ({_gname(ctx.group)})")
        return out, None


class _Wait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buf):
        work = _PENDING.pop(buf.data_ptr(), None)
        if work is not None:
            work.wait()                                  # enqueues the dependency on the current stream
        return buf.view_as(buf)

    @staticmethod
    def backward(ctx, grad):
        return grad


@torch.compiler.disable
def all_gather_rows_start(x, group):
    return _AllGatherRowsStart.apply(x, group)


@torch.compiler.disable
def wait_gather(buf):
    return _Wait.apply(buf)
