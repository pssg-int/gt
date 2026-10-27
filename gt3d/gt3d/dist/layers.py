"""Distributed building blocks on a Grid2D besides attention: sparse aggregation (GCN branch),
BatchNorm with global statistics, all-reduce with autograd, distributed cross-entropy / accuracy.
All dense inputs/outputs are the owned node chunk [chunk, ...]; padded rows (ids >= num_nodes)
are excluded from statistics and losses through `grid.valid_mask`."""
import os

import torch
import torch.distributed as dist
from torch_scatter import segment_csr

from gt3d.attention.backend_torch import build_csr
from gt3d.dist.grid2d import Grid2D, all_gather_rows, reduce_scatter_rows


_STEP = [0]


def set_step(i: int):
    """Training step counter; DistBatchNorm1d updates its running statistics at most once per step,
    so recomputation under activation checkpointing does not double-update them."""
    _STEP[0] = int(i)


class _AllReduceSum(torch.autograd.Function):
    """y = sum over ranks of x.  y is consumed independently on every rank, so the gradient of the
    TOTAL loss w.r.t. x_local is the sum over ranks of the upstream gradients: backward all-reduces g."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        y = x.clone()
        if not _NO_SYNC:
            dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous().clone()
        if not _NO_SYNC:
            dist.all_reduce(g, group=ctx.group)
        return g, None


# timing ablation only (wrong statistics): GT3D_BN_NO_SYNC=1 skips the BatchNorm statistics all-reduces
_NO_SYNC = os.environ.get("GT3D_BN_NO_SYNC", "0") == "1"


@torch.compiler.disable
def all_reduce_sum(x, group=None):
    return _AllReduceSum.apply(x, group)


class _SpMM(torch.autograd.Function):
    """out = A @ x with A a fixed sparse CSR matrix (cuSPARSE); backward dx = A^T @ dout using the
    pre-built transposed CSR.  Nothing of size [E, d] is ever materialised."""

    @staticmethod
    def forward(ctx, x, a_csr, at_csr):
        ctx.at_csr = at_csr
        return torch.sparse.mm(a_csr, x)

    @staticmethod
    def backward(ctx, dout):
        return torch.sparse.mm(ctx.at_csr, dout.contiguous()), None, None


class GCNBlock:
    """Normalised adjacency block A[r, c] shared by all GCN layers of a model (built once): CSR (row-local dst,
    col-local src) with the GCN weights, and its transpose (CSR of A^T, built by a stable sort on the source column),
    both handed to cuSPARSE as CSR matrices.  Memory: indices are int32 whenever the block allows it (always, per
    rank), the weights are kept in the first dtype the matrices are requested in (fp64 only until then) and the
    CSR -> input-order permutation is dropped unless keep_perm (only the full block needs it, to hand its
    normalised weights to the column sub-blocks / ring chunks)."""

    def __init__(self, grid, block_edge_index, edge_weight=None, n_cols=None, keep_perm=False):
        """grid: Grid2D (block rows = P_o chunks, columns = P_d chunks) or Grid3D (rows = P_o P_i chunks, columns =
        P_d P_i chunks: the block is replicated over the P_i ranks of a head group, each aggregating a d / P_i feature
        slice)."""
        self.grid = grid
        n_rows = grid.n_row_block
        c_lo, c_hi = grid.col_block_range()
        crow, col, _, perm = build_csr(block_edge_index, n_rows, None)
        self.n_rows, self.n_cols = n_rows, ((c_hi - c_lo) if n_cols is None else n_cols)
        rows = torch.repeat_interleave(torch.arange(n_rows, device=crow.device), crow[1:] - crow[:-1])
        if edge_weight is not None:                           # pre-normalised weights (e.g. from the partition files)
            w = edge_weight[perm].to(torch.float64)
        else:
            # in-degree of every row-block node = sum over the column blocks of the local counts (the row group spans
            # every column block; on a Grid3D the head-group ranks hold the same block, so the sum is still exact)
            deg = (crow[1:] - crow[:-1]).to(torch.float64)
            dist.all_reduce(deg, group=grid.row_group)
            dis_rows = deg.pow(-0.5)
            dis_rows[torch.isinf(dis_rows)] = 0.0
            pos = int(grid.global_to_row_local(torch.tensor([grid.owned_range()[0]])).item())   # my chunk inside the row block
            my_dis = dis_rows[pos:pos + grid.chunk].contiguous()
            p_i = getattr(grid, "p_heads", 1)
            if p_i > 1:                                       # column block = (r', h') chunks: gather the head group first
                hd = torch.empty(p_i * grid.chunk, dtype=torch.float64, device=crow.device)
                dist.all_gather_into_tensor(hd, my_dis, group=grid.head_group)
                my_dis = hd
            dis_cols = torch.empty(self.n_cols, dtype=torch.float64, device=crow.device)
            dist.all_gather_into_tensor(dis_cols, my_dis, group=grid.col_group)
            w = dis_rows[rows] * dis_cols[col]
        self._perm = perm if keep_perm else None
        perm_t = torch.argsort(col, stable=True)              # transpose: edges by source column, rows ascending
        ccol = torch.zeros(self.n_cols + 1, dtype=torch.long, device=crow.device)
        ccol[1:] = torch.cumsum(torch.bincount(col, minlength=self.n_cols), 0)
        idt = torch.int32 if (col.numel() < 2 ** 31 and max(self.n_rows, self.n_cols) < 2 ** 31) else torch.long
        self.crow, self.col = crow.to(idt).contiguous(), col.to(idt).contiguous()
        self.crow_t, self.col_t = ccol.to(idt).contiguous(), rows[perm_t].to(idt).contiguous()
        self.w, self.w_t = w, w[perm_t]                       # fp64 until the first matrices() call
        self._mats = {}
        self._cmats, self._rows_c, self._crow_c = {}, None, None

    def weights_in_input_order(self):
        """GCN-normalised edge weights aligned with the edge_index the block was built from (needs keep_perm)."""
        assert self._perm is not None, "GCNBlock was built without keep_perm"
        w = torch.empty_like(self.w)
        w[self._perm] = self.w
        return w

    def matrices(self, dtype):
        m = self._mats.get(dtype)
        if m is None:
            if not self._mats and dtype != self.w.dtype:      # keep the weights in the working dtype only
                self.w, self.w_t = self.w.to(dtype), self.w_t.to(dtype)
            a = torch.sparse_csr_tensor(self.crow, self.col, self.w.to(dtype), size=(self.n_rows, self.n_cols))
            at = torch.sparse_csr_tensor(self.crow_t, self.col_t, self.w_t.to(dtype), size=(self.n_cols, self.n_rows))
            m = self._mats[dtype] = (a, at)
        return m

    def matrices_compact(self, dtype):
        """(rows_c, A_c): the rows of A that own an edge (int64) and the CSR of those rows only (doubly compressed
        CSR, same columns and weights).  For the streamed layer's hypersparse column sub-blocks: the SpMM output is
        [len(rows_c), d] and is index_add_-ed onto the row-block accumulator instead of a full [n_rows, d] output
        plus a full-row add."""
        a_c = self._cmats.get(dtype)
        if a_c is None:
            if self._rows_c is None:
                deg = self.crow[1:] - self.crow[:-1]
                self._rows_c = torch.nonzero(deg > 0).squeeze(1)
                self._crow_c = torch.cat([self.crow[self._rows_c], self.crow[-1:]]).contiguous()
            a, _ = self.matrices(dtype)
            a_c = torch.sparse_csr_tensor(self._crow_c, self.col, a.values(), size=(self._rows_c.numel(), self.n_cols))
            self._cmats[dtype] = a_c
        return self._rows_c, a_c

    def nbytes(self, seen=None):
        """Bytes of the persistent structures (unique storages)."""
        ts = [self.crow, self.col, self.crow_t, self.col_t, self.w, self.w_t, self._perm, self._rows_c, self._crow_c]
        ts += [v for a, at in self._mats.values() for v in (a.values(), at.values())]
        seen = set() if seen is None else seen
        total = 0
        for t in ts:
            if t is not None and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); total += t.numel() * t.element_size()
        return total


class GCNBlockView:
    """A GCNBlock restricted to the compact row / column index spaces of a sparse-communication plan pair
    (gt3d.dist.sparse_comm): rows_need must contain every non-empty row of the block and col_remap must map every
    referenced column (both hold when the need sets are taken from the union of the attention and GCN tiles).
    Same weights as the block; same interface (matrices / matrices_compact / nbytes)."""

    def __init__(self, block: GCNBlock, rows_need, n_rows_c, col_remap, n_cols_c, cols_need=None):
        from gt3d.dist.sparse_comm import compact_csr
        self.block, self.n_rows, self.n_cols = block, int(n_rows_c), int(n_cols_c)
        crow_c, col_c = compact_csr(block.crow, rows_need, block.col, col_remap)
        idt = torch.int32 if (col_c.numel() < 2 ** 31 and max(self.n_rows, self.n_cols) < 2 ** 31) else torch.long
        self.crow, self.col = crow_c.to(idt).contiguous(), col_c.to(idt).contiguous()
        # transpose: the block's CSC rows are the full column block; keep the referenced columns only
        if cols_need is None:
            cols_need = torch.nonzero(col_remap >= 0).squeeze(1)
        assert cols_need.numel() == self.n_cols, (cols_need.numel(), self.n_cols)
        rows_remap = torch.full((block.n_rows,), -1, dtype=torch.long, device=block.crow.device)
        rows_remap[rows_need.long()] = torch.arange(self.n_rows, device=block.crow.device)
        crow_t_c, col_t_c = compact_csr(block.crow_t, cols_need, block.col_t, rows_remap)
        self.crow_t, self.col_t = crow_t_c.to(idt).contiguous(), col_t_c.to(idt).contiguous()
        self._rows_remap = rows_remap
        self._mats, self._cmats, self._rows_c, self._crow_c = {}, {}, None, None

    def matrices(self, dtype):
        m = self._mats.get(dtype)
        if m is None:
            a_full, at_full = self.block.matrices(dtype)
            a = torch.sparse_csr_tensor(self.crow, self.col, a_full.values(), size=(self.n_rows, self.n_cols))
            at = torch.sparse_csr_tensor(self.crow_t, self.col_t, at_full.values(), size=(self.n_cols, self.n_rows))
            m = self._mats[dtype] = (a, at)
        return m

    def matrices_compact(self, dtype):
        a_c = self._cmats.get(dtype)
        if a_c is None:
            if self._rows_c is None:
                rows_c, crow_c = self.block.matrices_compact(dtype)[0], None
                self._rows_c = self._rows_remap[rows_c.long()]
                assert bool((self._rows_c >= 0).all())
                self._crow_c = self.block._crow_c
            a, _ = self.matrices(dtype)
            a_c = torch.sparse_csr_tensor(self._crow_c, self.col, a.values(), size=(self._rows_c.numel(), self.n_cols))
            self._cmats[dtype] = a_c
        return self._rows_c, a_c

    def nbytes(self, seen=None):
        seen = set() if seen is None else seen
        total = self.block.nbytes(seen)
        for t in (self.crow, self.col, self.crow_t, self.col_t, self._rows_c, self._crow_c):
            if t is not None and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); total += t.numel() * t.element_size()
        return total


@torch.compiler.disable
def spmm(x, a, at):
    return _SpMM.apply(x, a, at)


class DistGCNConv(torch.nn.Module):
    """PyG GCNConv(dim, dim) (symmetric normalisation, one self loop per node) on the local block
    of the ORIGINAL graph: X W computed on the owned chunk, all-gathered over the column group,
    aggregated on the block (cuSPARSE SpMM), reduce-scattered over the row group, plus bias."""

    def __init__(self, grid: Grid2D, block: GCNBlock, dim_in, dim_out):
        super().__init__()
        self.grid, self.block = grid, block
        self.lin = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(dim_out))

    def forward(self, x_loc):
        h = self.lin(x_loc)                                   # owned chunk
        h_blk = all_gather_rows(h, self.grid.col_group)      # column block
        a, at = self.block.matrices(h.dtype)
        agg = spmm(h_blk, a, at)                             # [n_rows, d], no per-edge temporaries
        out = reduce_scatter_rows(agg, self.grid.row_group)  # owned chunk
        return out + self.bias


class DistBatchNorm1d(torch.nn.Module):
    """BatchNorm1d whose batch statistics are taken over all valid nodes of all ranks."""

    def __init__(self, grid: Grid2D, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.grid, self.eps, self.momentum = grid, eps, momentum
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self._stats_step = -1

    def local_stats(self, x):
        """Masked local sums [2*dim]: (sum x, sum x^2) over the valid owned rows."""
        mask = self.grid.valid_mask.to(x.dtype).unsqueeze(-1)
        xm = x * mask
        return torch.cat([xm.sum(0), (xm * x).sum(0)])

    def apply_stats(self, x, stats):
        """Normalise with the GLOBAL sums `stats` [2*dim] (already all-reduced); updates running stats once per step."""
        n = float(self.grid.num_nodes)
        s, ss = stats[: x.shape[1]], stats[x.shape[1]:]
        mean = s / n
        var = ss / n - mean * mean                             # biased variance (as torch uses for normalisation)
        if self._stats_step != _STEP[0]:
            self._stats_step = _STEP[0]
            with torch.no_grad():
                unbiased = var * n / max(n - 1.0, 1.0)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach().to(self.running_mean.dtype))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased.detach().to(self.running_var.dtype))
                self.num_batches_tracked += 1
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight.to(x.dtype) + self.bias.to(x.dtype)

    def forward(self, x):
        if self.training:
            return self.apply_stats(x, all_reduce_sum(self.local_stats(x)))    # one collective for both moments
        mean, var = self.running_mean.to(x.dtype), self.running_var.to(x.dtype)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight.to(x.dtype) + self.bias.to(x.dtype)


def dist_batchnorm_pair(bn_a, x_a, bn_b, x_b):
    """Two independent DistBatchNorm1d applications sharing ONE statistics all-reduce (and one in backward):
    used for the GCN and attention branches of an Exphormer layer."""
    if not bn_a.training:
        return bn_a(x_a), bn_b(x_b)
    d = x_a.shape[1]
    stats = all_reduce_sum(torch.cat([bn_a.local_stats(x_a), bn_b.local_stats(x_b)]))
    return bn_a.apply_stats(x_a, stats[: 2 * d]), bn_b.apply_stats(x_b, stats[2 * d:])


def dist_cross_entropy(logits_loc, y_loc, mask_loc, num_total):
    """Mean cross-entropy over the masked nodes of all ranks (each rank returns its share; the
    parameter gradients must be summed across ranks afterwards)."""
    if mask_loc.any():
        loss = torch.nn.functional.cross_entropy(logits_loc[mask_loc], y_loc[mask_loc], reduction="sum")
    else:
        loss = logits_loc.sum() * 0.0
    return loss / num_total


@torch.no_grad()
def dist_accuracy(logits_loc, y_loc, masks_loc: dict):
    """Accuracy per split over all ranks. masks_loc: name -> bool mask over the owned chunk."""
    pred = logits_loc.argmax(-1)
    stats = torch.stack([torch.stack([(pred[m] == y_loc[m]).sum(), m.sum()]) for m in masks_loc.values()]).double()
    dist.all_reduce(stats)
    return {k: (stats[i, 0] / stats[i, 1].clamp_min(1)).item() for i, k in enumerate(masks_loc)}


def allreduce_grads(params, group=None):
    """Flat sum-all-reduce of all gradients (replicated parameters).  Parameters that received no
    gradient on this rank (e.g. the global token, used by one rank only) contribute zeros, so every
    rank reduces the same flat buffer."""
    params = [p for p in params if p.requires_grad]
    if not params:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat, group=group)
    off = 0
    for p in params:
        p.grad.copy_(flat[off:off + p.numel()].view_as(p))
        off += p.numel()
