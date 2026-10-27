"""3D (rows x columns x heads) process grid: the 2D block grid of Grid2D plus a DeepSpeed-Ulysses-style head axis.

P = P_d * P_o * P_i ranks, rank = (r * P_o + c) * P_i + h (head index fastest: the P_i ranks of a head group are
consecutive, i.e. inside one node for P_i <= GPUs per node).

Ownership (node-sharded layout, every rank owns one chunk of N_pad / P nodes with ALL features / heads):
  rank (r, c, h) owns chunk index  (c * P_d + r) * P_i + h.
  * column block c = chunks of the ranks (., c, .) = ids [c*P_d*P_i*chunk, (c+1)*P_d*P_i*chunk)  -> contiguous
  * row block r    = chunks of the ranks (r, ., .), in row-local order (c', h')                  -> P_o*P_i chunks
Rank (r, c, h) computes the attention block A[row block r, column block c] for head group h
(heads [h*H/P_i, (h+1)*H/P_i)).  Process groups:
  * head group (r, c, .) : all-to-all rows <-> heads (Ulysses), P_i ranks
  * col  group (., c, h) : all-gather of the head slice of K, V for column block c, P_d ranks
  * row  group (r, ., h) : all-gather of the head slice of Q for row block r, reduce-scatter of the partials, P_o ranks
(P_d, P_o, 1) is the 2D grid with projected (Q, K, V) gathers (no redundant projections) instead of the fused
layer's x gathers; (1, 1, P) is pure head parallelism = DeepSpeed-Ulysses / TorchGT / DAC'26 GP-A2A: the whole
graph on every rank, P <= H.
"""
import torch
import torch.distributed as dist


class Grid3D:
    def __init__(self, p_rows: int, p_cols: int, p_heads: int, num_nodes: int, n_pad: int = None, timers: bool = False,
                 use_axonn: bool = None):
        """Process groups are torch.distributed groups (every rank creates every group).  With the NCCL backend AxoNN is
        initialised as well (G_intra_c = P_i fastest, G_intra_r = P_o, G_intra_d = P_d: the same rank arrangement), so
        that its timers (`timers`, ax.get_timers()) and the Plexus helpers work exactly as on a Grid2D; its groups are
        not used."""
        assert dist.is_initialized()
        world = dist.get_world_size()
        assert world == p_rows * p_cols * p_heads, (world, p_rows, p_cols, p_heads)
        self.p_rows, self.p_cols, self.p_heads = p_rows, p_cols, p_heads
        self.rank = dist.get_rank()
        rc, self.h = divmod(self.rank, p_heads)
        self.r, self.c = divmod(rc, p_cols)
        self.num_nodes = num_nodes
        if use_axonn is None:
            use_axonn = dist.get_backend() == "nccl"
        self.use_axonn = use_axonn
        if use_axonn:
            from axonn import axonn as ax
            if not ax.is_initialized:
                ax.init(G_inter=1, G_data=1, G_intra_r=p_cols, G_intra_c=p_heads, G_intra_d=p_rows, enable_internal_timers=timers)
            elif timers:
                ax.enable_timers = True
            ax_col = self.rank % p_heads
            ax_row = (self.rank // p_heads) % p_cols
            ax_depth = self.rank // (p_heads * p_cols)
            assert (ax_depth, ax_row, ax_col) == (self.r, self.c, self.h)
        if n_pad is None:
            self.chunk = -(-num_nodes // world)
        else:
            assert n_pad % world == 0 and n_pad >= num_nodes, (n_pad, world, num_nodes)
            self.chunk = n_pad // world
        self.n_pad = self.chunk * world
        rank_of = lambda r, c, h: (r * p_cols + c) * p_heads + h   # noqa: E731
        self.rank_of = rank_of
        # torch.distributed requires every rank to create every group
        self.head_groups = {(r, c): dist.new_group([rank_of(r, c, h) for h in range(p_heads)])
                            for r in range(p_rows) for c in range(p_cols)}
        self.col_groups = {(c, h): dist.new_group([rank_of(r, c, h) for r in range(p_rows)])
                           for c in range(p_cols) for h in range(p_heads)}
        self.row_groups = {(r, h): dist.new_group([rank_of(r, c, h) for c in range(p_cols)])
                           for r in range(p_rows) for h in range(p_heads)}
        self.head_group = self.head_groups[(self.r, self.c)]
        self.col_group = self.col_groups[(self.c, self.h)]
        self.row_group = self.row_groups[(self.r, self.h)]
        from gt3d.dist.grid2d import GROUP_NAMES
        GROUP_NAMES[id(self.row_group)] = "row"
        GROUP_NAMES[id(self.col_group)] = "col"
        GROUP_NAMES[id(self.head_group)] = "head"
        lo, hi = self.owned_range()
        self.valid_mask = torch.arange(lo, hi) < num_nodes

    def to(self, device):
        self.valid_mask = self.valid_mask.to(device)
        return self

    def take_owned(self, x_global):
        lo, hi = self.owned_range()
        out = x_global.new_zeros((self.chunk,) + tuple(x_global.shape[1:]))
        n = min(hi, self.num_nodes) - lo
        if n > 0:
            out[:n] = x_global[lo:lo + n]
        return out

    # ---- ownership ----------------------------------------------------------------------
    def chunk_of(self, r, c, h):
        return (c * self.p_rows + r) * self.p_heads + h

    def owned_range(self, r=None, c=None, h=None):
        r = self.r if r is None else r
        c = self.c if c is None else c
        h = self.h if h is None else h
        k = self.chunk_of(r, c, h)
        return k * self.chunk, (k + 1) * self.chunk

    def owner_of(self, node_ids):
        """(r, c, h) of the owner of each (padded) node id."""
        k = node_ids // self.chunk
        h = k % self.p_heads
        rc = k // self.p_heads
        return rc % self.p_rows, rc // self.p_rows, h

    def col_block_range(self, c=None):
        c = self.c if c is None else c
        size = self.p_rows * self.p_heads * self.chunk
        return c * size, (c + 1) * size

    def row_block_ranges(self, r=None):
        """(lo, hi) global id ranges of the chunks of row block r, in row-local order (c', h')."""
        r = self.r if r is None else r
        return [self.owned_range(r, c, h) for c in range(self.p_cols) for h in range(self.p_heads)]

    @property
    def n_row_block(self):
        return self.p_cols * self.p_heads * self.chunk

    def row_block_ids(self, r=None):
        return torch.cat([torch.arange(lo, hi) for lo, hi in self.row_block_ranges(r)])

    def global_to_row_local(self, gid):
        r, c, h = self.owner_of(gid)
        return (c * self.p_heads + h) * self.chunk + (gid - self.chunk_of(r, c, h) * self.chunk)

    def global_to_col_local(self, gid):
        lo, _ = self.col_block_range(self.owner_of(gid)[1])
        return gid - lo

    def local_block(self, edge_index, edge_type=None):
        src, dst = edge_index[0], edge_index[1]
        r_dst = self.owner_of(dst)[0]
        c_src = self.owner_of(src)[1]
        m = (r_dst == self.r) & (c_src == self.c)
        ei = torch.stack([self.global_to_col_local(src[m]), self.global_to_row_local(dst[m])])
        return ei, (edge_type[m] if edge_type is not None else None)


# ---- Ulysses all-to-all (rows <-> heads) with autograd ---------------------------------------
def _a2a(x, group):
    from gt3d.dist.grid2d import _pre_collective
    _pre_collective(group)
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x.contiguous(), group=group)
    return out


def raw_rows_to_heads(x, group, p):
    """[chunk, H, d] (my rows, all heads) -> [p*chunk, H/p, d] (the p chunks of my head group, my heads),
    rows in head-group rank order = contiguous global ids."""
    if p == 1:
        return x
    chunk, h, d = x.shape
    send = x.view(chunk, p, h // p, d).transpose(0, 1).contiguous()      # piece j = my rows, head group j -> rank j
    return _a2a(send, group).view(p * chunk, h // p, d)                   # piece i = rows of rank i, my head group


def raw_heads_to_rows(y, group, p):
    """[p*chunk, H/p, d] -> [chunk, H, d]: the exact inverse of raw_rows_to_heads."""
    if p == 1:
        return y
    n, hg, d = y.shape
    chunk = n // p
    recv = _a2a(y.contiguous().view(p, chunk, hg, d), group)              # piece j = my rows, head group j
    return recv.transpose(0, 1).contiguous().view(chunk, p * hg, d)


class _RowsToHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, p):
        ctx.group, ctx.p = group, p
        return raw_rows_to_heads(x, group, p)

    @staticmethod
    def backward(ctx, grad):
        from gt3d.dist.grid2d import _timers
        t = _timers(); t.start("bwd all-to-all (head)")
        out = raw_heads_to_rows(grad, ctx.group, ctx.p)
        t.stop("bwd all-to-all (head)")
        return out, None, None


class _HeadsToRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, group, p):
        ctx.group, ctx.p = group, p
        return raw_heads_to_rows(y, group, p)

    @staticmethod
    def backward(ctx, grad):
        from gt3d.dist.grid2d import _timers
        t = _timers(); t.start("bwd all-to-all (head)")
        out = raw_rows_to_heads(grad, ctx.group, ctx.p)
        t.stop("bwd all-to-all (head)")
        return out, None, None


@torch.compiler.disable
def rows_to_heads(x, group, p):
    return _RowsToHeads.apply(x, group, p)


@torch.compiler.disable
def heads_to_rows(y, group, p):
    return _HeadsToRows.apply(y, group, p)
