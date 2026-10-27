"""Distributed Exphormer (GCN+Exphormer MultiModel) on a Grid2D or on a Grid3D (2D blocks x head axis);
parameters and semantics are identical to gt3d.models.exphormer_ref.ExphormerRef, so its state dict (and the
official one via load_official_state_dict) can be loaded directly for numerical tests."""
import torch
import torch.utils.checkpoint
import torch.distributed as dist
import os

# GT3D_BN_PAIR=0 restores one all-reduce per BatchNorm (the GCN and attention BatchNorms of a layer are independent)
BN_PAIR = os.environ.get("GT3D_BN_PAIR", "1") == "1"


def _timers():
    """AxoNN's CUDA-event timers (no-ops unless enabled via Grid2D(timers=True) / --timers)."""
    from axonn import axonn as ax
    return ax.get_timers()
NORM_KIND = ["bn"]                 # set by DistExphormer(norm=...) before the layers are built
import torch.nn as nn
import torch.nn.functional as F

from gt3d.dist.grid2d import Grid2D
from gt3d.dist.layers import DistGCNConv, DistBatchNorm1d, GCNBlock, dist_batchnorm_pair
from gt3d.dist.sparse_attention import DistMaskedSparseAttention, AttentionBlock


def gcn_edges_with_self_loops(edge_index, num_nodes):
    """PyG add_remaining_self_loops semantics on a global edge list: drop all self loops, add one per node."""
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    loops = torch.arange(num_nodes, device=edge_index.device)
    return torch.stack([torch.cat([src[keep], loops]), torch.cat([dst[keep], loops])])


def make_norm(norm, grid, dim, bn_eps, bn_momentum):
    """'bn': DistBatchNorm1d (global statistics, one all-reduce per call); 'ln': LayerNorm per node (no collective);
    'rms': Plexus's ScatteredRMSNorm (per node, no collective; the norm used by Plexus's scattered GCN)."""
    if norm == "ln":
        return nn.LayerNorm(dim, eps=bn_eps)
    if norm == "rms":
        from plexus.scattered import ScatteredRMSNorm
        return ScatteredRMSNorm(dim, eps=bn_eps)
    return DistBatchNorm1d(grid, dim, bn_eps, bn_momentum)


def _is_bn(mod):
    return isinstance(mod, DistBatchNorm1d)


class DistExphormerLayer(nn.Module):
    """with_gcn=False: attention-only layer (GraphGPS local_gnn_type=None): h = BN(x + Attn(x)) -> FFN block; no GCN
    parameters are created.  with_attn=False: GCN-only layer (GraphGPS global_model_type=None): h = BN(x + GCN(x)) -> FFN
    block; no attention parameters are created."""

    def __init__(self, grid, gcn_block: GCNBlock, attn_block: AttentionBlock, dim_h, num_heads, dropout,
                 bn_eps, bn_momentum, backend, comm_dtype=None, with_gcn=True, with_attn=True):
        super().__init__()
        assert with_gcn or with_attn
        self.dim_h, self.num_heads, self.d_head = dim_h, num_heads, dim_h // num_heads
        self.dropout = dropout
        self.with_gcn, self.with_attn = with_gcn, with_attn
        if with_gcn:
            self.gcn = DistGCNConv(grid, gcn_block, dim_h, dim_h)
            self.bn_local = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        if with_attn:
            self.q = nn.Linear(dim_h, dim_h, bias=False)
            self.k = nn.Linear(dim_h, dim_h, bias=False)
            self.v = nn.Linear(dim_h, dim_h, bias=False)
            self.e = nn.Linear(dim_h, dim_h, bias=False)
            self.attn = DistMaskedSparseAttention(grid, attn_block, num_heads, self.d_head,
                                                  backend=backend, score="gated", norm="clampexp", comm_dtype=comm_dtype)
            self.bn_attn = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        self.ff1 = nn.Linear(dim_h, 2 * dim_h)
        self.ff2 = nn.Linear(2 * dim_h, dim_h)
        self.bn_ff = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)

    def forward(self, x, edge_type_emb):
        n, h, d = x.shape[0], self.num_heads, self.d_head
        if self.with_gcn:
            h_local = self.gcn(x)
            h_local = F.dropout(h_local, self.dropout, training=self.training)
            h_local = self.bn_local(x + h_local)
        if self.with_attn:
            gate = self.e(edge_type_emb).view(-1, h, d)
            o = self.attn(self.q(x).view(n, h, d), self.k(x).view(n, h, d), self.v(x).view(n, h, d), gate)
            h_attn = F.dropout(o.reshape(n, h * d), self.dropout, training=self.training)
            h_attn = self.bn_attn(x + h_attn)

        hh = (h_local + h_attn) if (self.with_gcn and self.with_attn) else (h_attn if self.with_attn else h_local)
        ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(hh + ff)


class DistExphormer(nn.Module):
    def __init__(self, grid: Grid2D, gcn_block_ei, attn_block_ei, attn_block_et, dim_in, dim_hidden, dim_out,
                 num_layers, num_heads, dropout=0.3, gnn_dropout=0.3, num_edge_types=2, bn_eps=1e-5,
                 bn_momentum=0.1, pre_l2norm=True, backend="triton", gcn_weight=None, checkpoint=False, comm_dtype=None,
                 fused=False, rs_dtype=None, pipeline_sub=1, stream_sub=1, norm="bn", ring=False, with_gcn=True,
                 with_attn=True, gather_proj=False, stream_overlap=False, stream_compact=True, kernel_dtype=None,
                 sparse_comm=False, sparse_threshold=None):
        """sparse_comm: sparsity-aware gathers / reduce-scatters (gt3d.dist.sparse_comm; fused, streamed and head
        layers): every rank receives only the rows of the row / column block its tile references, over an
        all-to-all-v with static index sets.  sparse_threshold: forced share limit per collective (1.0 = always sparse,
        0 = never); None = the measured a2a / all-gather rate of the group's node layout x margin."""
        super().__init__()
        self.grid = grid
        NORM_KIND[0] = norm
        self.sparse_comm, self.sparse_info = None, None
        assert not gather_proj or (fused and max(pipeline_sub, stream_sub) == 1 and not ring), \
            "gather_proj is implemented for the fused (non-streamed) layer"
        self.with_gcn, self.with_attn = with_gcn, with_attn
        assert with_gcn or max(pipeline_sub, stream_sub) == 1, \
            "attention-only (with_gcn=False) is implemented for the plain, fused, head-grid and ring layers"
        assert with_attn or (not fused and not ring), "GCN-only (with_attn=False) is implemented for the plain layer"
        head_grid = hasattr(grid, "p_heads")                            # Grid3D: 2D blocks x Ulysses head axis
        if head_grid:
            assert with_attn and not fused and not ring and max(pipeline_sub, stream_sub) == 1 and not gather_proj, \
                "the head grid has its own layer (DistExphormerLayerHead): no fused / ring / streamed / gather_proj"
        self.gnn_dropout, self.pre_l2norm, self.checkpoint = gnn_dropout, pre_l2norm, checkpoint
        n_sub = max(pipeline_sub, stream_sub)
        gcn_block = GCNBlock(grid, gcn_block_ei, gcn_weight,          # graph structures shared by all layers
                             keep_perm=ring or (fused and n_sub > 1)) if with_gcn else None   # perm only needed to split the weights
        attn_block = AttentionBlock(grid, attn_block_ei, attn_block_et)
        if sparse_comm:
            assert with_attn and not ring and pipeline_sub <= 1 and (fused or head_grid), \
                "sparse communication is implemented for the fused, streamed and head-grid layers"
        self.edge_type_emb = nn.Embedding(num_edge_types, dim_hidden)
        self.pre_lin = nn.Linear(dim_in, dim_hidden, bias=False)
        self.pre_bn = make_norm(NORM_KIND[0], grid, dim_hidden, bn_eps, bn_momentum)
        extra = dict(rs_dtype=rs_dtype, kernel_dtype=kernel_dtype) if fused else {}
        if gather_proj:
            extra["gather_proj"] = True
        if not with_gcn:
            extra["with_gcn"] = False
        if not with_attn:
            extra["with_attn"] = False
        if head_grid:
            sp = None
            if sparse_comm:
                sp, self.sparse_info = build_sparse_comm(grid, attn_block, gcn_block, sparse_threshold)
                if sp is not None:
                    attn_block, gcn_block, self.sparse_comm = sp["attn"], sp["gcn"], sp
            self.layers = nn.ModuleList([
                DistExphormerLayerHead(grid, attn_block, dim_hidden, num_heads, dropout, bn_eps, bn_momentum, backend,
                                       comm_dtype=comm_dtype, rs_dtype=rs_dtype, kernel_dtype=kernel_dtype,
                                       gcn_block=gcn_block, with_gcn=with_gcn, sparse=sp)
                for _ in range(num_layers)])
        elif ring:
            # ring attention (gt3d/dist/ring.py): (P, 1) grid, one sub-block per source chunk = per ring step
            from gt3d.dist.ring import DistExphormerLayerRing, split_block_by_source_chunk
            assert grid.p_cols == 1, "ring attention needs a (P, 1) grid"
            p = grid.p_rows
            if with_gcn:
                gcn_w_full = gcn_block.weights_in_input_order()           # degrees of the full row block
                gcn_parts = split_block_by_source_chunk(gcn_block_ei, p, grid.chunk, None, gcn_w_full)
                gcn_blocks = [GCNBlock(grid, ei, w, n_cols=grid.chunk) for ei, _, w in gcn_parts]
            else:                                                         # attention-only ring: no GCN structures
                gcn_blocks = [None] * p
            attn_parts = split_block_by_source_chunk(attn_block_ei, p, grid.chunk, attn_block_et, None)
            del gcn_block, attn_block
            attn_blocks = [AttentionBlock(grid, ei, et) for ei, et, _ in attn_parts]
            self.layers = nn.ModuleList([
                DistExphormerLayerRing(grid, gcn_blocks, attn_blocks, dim_hidden, num_heads, dropout, bn_eps, bn_momentum,
                                       backend, comm_dtype=comm_dtype, tail_checkpoint=checkpoint, with_gcn=with_gcn)
                for _ in range(num_layers)])
        elif fused and n_sub > 1:
            # per-column-sub-block structures (shared by all layers)
            n_cols_sub = grid.p_rows * (grid.chunk // n_sub)
            # normalisation needs the FULL block's degrees: take the full block's weights, then split
            gcn_w_full = gcn_block.weights_in_input_order()
            gcn_parts = split_block_by_column_subblocks(gcn_block_ei, n_sub, grid.chunk, grid.p_rows, None, gcn_w_full)
            attn_parts = split_block_by_column_subblocks(attn_block_ei, n_sub, grid.chunk, grid.p_rows, attn_block_et, None)
            gcn_blocks = [GCNBlock(grid, ei, w, n_cols=n_cols_sub) for ei, _, w in gcn_parts]
            attn_blocks = [AttentionBlock(grid, ei, et) for ei, et, _ in attn_parts]
            sp = None
            if sparse_comm:
                sp, self.sparse_info = build_sparse_comm(grid, attn_block, gcn_block, sparse_threshold,
                                                         attn_subs=attn_blocks, gcn_subs=gcn_blocks)
                if sp is not None:
                    attn_blocks, gcn_blocks, self.sparse_comm = sp["attn"], sp["gcn"], sp
            del attn_block
            if stream_sub > 1:
                self.layers = nn.ModuleList([
                    DistExphormerLayerStreamed(grid, gcn_blocks, attn_blocks, dim_hidden, num_heads, dropout,
                                               bn_eps, bn_momentum, backend, comm_dtype=comm_dtype, rs_dtype=rs_dtype,
                                               tail_checkpoint=checkpoint, kernel_dtype=kernel_dtype)
                    for _ in range(num_layers)])
                for layer in self.layers:
                    layer.overlap = stream_overlap
                    layer.compact = stream_compact       # compact-accumulate attention kernels (triton backend)
                    layer.sp_row = sp["row"] if sp is not None else None
                    layer.sp_cols = sp["cols"] if sp is not None else None
            else:
                self.layers = nn.ModuleList([
                    DistExphormerLayerPipelined(grid, gcn_blocks, attn_blocks, dim_hidden, num_heads, dropout,
                                                bn_eps, bn_momentum, backend, comm_dtype=comm_dtype, rs_dtype=rs_dtype,
                                                kernel_dtype=kernel_dtype)
                    for _ in range(num_layers)])
        else:
            layer_cls = DistExphormerLayerFused if fused else DistExphormerLayer
            if sparse_comm:
                sp, self.sparse_info = build_sparse_comm(grid, attn_block, gcn_block, sparse_threshold)
                if sp is not None:
                    attn_block, gcn_block, self.sparse_comm = sp["attn"], sp["gcn"], sp
                    extra["sparse"] = sp
            self.layers = nn.ModuleList([
                layer_cls(grid, gcn_block, attn_block, dim_hidden, num_heads, dropout,
                          bn_eps, bn_momentum, backend, comm_dtype=comm_dtype, **extra) for _ in range(num_layers)])
        self.head = nn.Linear(dim_hidden, dim_out, bias=True)

    def forward(self, x_loc):
        h = self.pre_lin(x_loc)
        h = self.pre_bn(h)
        h = F.dropout(h, self.gnn_dropout, training=self.training)
        h = F.relu(h)
        if self.pre_l2norm:
            h = F.normalize(h, p=2, dim=1)
        for layer in self.layers:
            if self.checkpoint and self.training and not getattr(layer, "self_checkpointing", False):
                # recompute the layer in backward; BatchNorm running stats update once per step (see set_step)
                h = torch.utils.checkpoint.checkpoint(layer, h, self.edge_type_emb.weight, use_reentrant=False)
            else:
                h = layer(h, self.edge_type_emb.weight)
        return self.head(h)

    def compaction_stats(self):
        """(row-block rows x attention blocks, (row, block) pairs that own an edge) of the first layer: the share of
        the row block a column (sub-)block touches, i.e. what the compact-accumulate kernels tile."""
        attns = getattr(self.layers[0], "attns", None)
        if attns is None:
            return 0, 0
        tot = nonempty = 0
        for blk in attns:
            deg = blk.crow[1:] - blk.crow[:-1]
            tot += deg.numel()
            nonempty += int((deg > 0).sum())
        return tot, nonempty

    def structure_bytes(self):
        """Bytes of the graph structures (GCN / attention blocks and the plans cached on them) held by the model."""
        from gt3d.dist.layers import GCNBlock, GCNBlockView
        from gt3d.dist.sparse_attention import AttentionBlock, structure_tensor_bytes
        from gt3d.dist.sparse_comm import SparseRowPlan
        seen, total = set(), 0           # seen: storages (data_ptr) already counted

        def walk(obj, depth):
            nonlocal total
            if isinstance(obj, (GCNBlock, GCNBlockView, AttentionBlock)):
                total += obj.nbytes(seen)
            elif isinstance(obj, SparseRowPlan):
                total += obj.nbytes(seen)
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v, depth + 1)
            elif depth < 4:
                if isinstance(obj, (list, tuple, nn.ModuleList)):
                    for v in obj:
                        walk(v, depth + 1)
                elif isinstance(obj, nn.Module):
                    if hasattr(obj, "crow") and hasattr(obj, "col"):      # attention modules keep copies of the block's tensors
                        total += structure_tensor_bytes([obj.crow, obj.col, getattr(obj, "etype", None)], seen)
                    for v in list(vars(obj).values()) + list(obj._modules.values()):
                        walk(v, depth + 1)
        walk(self, 0)
        return total

    @torch.no_grad()
    def load_ref_state_dict(self, sd):
        """Load an ExphormerRef state dict (same parameter names; DistGCNConv uses lin/bias like GCNConvRef)."""
        own = self.state_dict()
        for k, v in sd.items():
            assert k in own, k
            own[k].copy_(v)
        self.load_state_dict(own)


# =============================================================================================
# Fused-communication layer: gather x once per block, project locally, one reduce-scatter.
#   forward : x_col = gather(x, col group)  [used by K, V and the GCN projection]
#             x_row = gather(x, row group)  [used by Q]
#             partials (acc, l) of A[r,c] + GCN aggregation partial -> ONE reduce-scatter
#   backward: autograd -> one all-gather of (dacc, dl, dagg), kernel backward, dx_col and dx_row each
#             reduce-scattered once (the gather wrappers' backward).
# 3 collectives per direction instead of 5-6, and no separate (o, dO) gather.
# =============================================================================================
from gt3d.dist.grid2d import all_gather_rows as _all_gather_rows, reduce_scatter_rows as _reduce_scatter_rows, all_gather_rows_start, wait_gather, raw_all_gather_rows, raw_reduce_scatter_rows
from gt3d.dist.layers import spmm
from gt3d.dist.sparse_attention import DistAttentionPartial, combine_partials, split_block_by_column_subblocks


def _gather_rows(t, group, plan=None):
    """all-gather over `group` (autograd: reduce-scatter), or the plan's sparse gather when a SparseRowPlan is given."""
    if plan is None:
        return _all_gather_rows(t, group)
    from gt3d.dist.sparse_comm import sparse_all_gather
    return sparse_all_gather(t, plan)


def build_sparse_comm(grid, attn_block, gcn_block, threshold=None, attn_subs=None, gcn_subs=None):
    """Sparse-communication plans (gt3d.dist.sparse_comm.SparseRowPlan) of the row group (Q-side gather, partials
    reduce-scatter) and the column group (K / V-side gather, dx reduce-scatter) for this rank's tile, and the tiles
    remapped to the plans' compact index spaces.  The need sets are the union of the attention and GCN tiles (the GCN
    edges are a subset of the attention edges, so the union costs nothing).  A plan is used only when its share of
    rows is below the group's a2a / all-gather rate x margin (or the forced `threshold`); the other collective stays
    dense (its index space is then the full block).  attn_subs / gcn_subs: the column sub-blocks of the streamed layer
    (one column plan per sub-block, rows_per_rank = chunk / n_sub).
    Returns (dict(row=, col=, cols=, attn=, gcn=, info=) or None, info) with info the shares / decisions for the log."""
    from gt3d.dist.sparse_comm import SparseRowPlan, tile_need_sets
    from gt3d.dist.layers import GCNBlockView
    dev = attn_block.crow.device
    n_rows = grid.n_row_block
    c_lo, c_hi = grid.col_block_range()
    n_cols = c_hi - c_lo
    g_row, g_col = dist.get_world_size(grid.row_group), dist.get_world_size(grid.col_group)

    def need(a_blk, g_blk, nc):
        rows, cols = tile_need_sets(a_blk.crow, a_blk.col, nc)
        if g_blk is not None:
            r2, c2 = tile_need_sets(g_blk.crow, g_blk.col, nc)
            rows, cols = torch.unique(torch.cat([rows, r2])), torch.unique(torch.cat([cols, c2]))
        return rows, cols

    rows, cols = need(attn_block, gcn_block, n_cols)
    plan_row = SparseRowPlan(grid.row_group, n_rows // g_row, rows, name="row")
    plan_col = SparseRowPlan(grid.col_group, n_cols // g_col, cols, name="col")
    thr_row, thr_col = threshold if isinstance(threshold, (tuple, list)) else (threshold, threshold)   # (row, col) or one for both
    use_row, use_col = plan_row.worthwhile(thr_row), plan_col.worthwhile(thr_col)
    info = dict(row_share=plan_row.share, col_share=plan_col.share, row_ratio=plan_row.ratio, col_ratio=plan_col.ratio,
                row_rows=plan_row.n_need, col_rows=plan_col.n_need, use_row=use_row, use_col=use_col,
                n_row_block=n_rows, n_col_block=n_cols)
    if not (use_row or use_col):
        return None, info
    rows_eff = rows if use_row else torch.arange(n_rows, device=dev)
    n_rows_c = int(rows_eff.numel())
    out = dict(row=plan_row if use_row else None, col=None, cols=None, info=info)
    if attn_subs is None:
        out["col"] = plan_col if use_col else None
        col_remap = plan_col.remap.long() if use_col else torch.arange(n_cols, device=dev)
        cols_eff = plan_col.need_local.long() if use_col else torch.arange(n_cols, device=dev)
        out["attn"] = attn_block.compacted(rows_eff, col_remap)
        out["gcn"] = (GCNBlockView(gcn_block, rows_eff, n_rows_c, col_remap, cols_eff.numel(), cols_eff)
                      if gcn_block is not None else None)
        del col_remap
    else:
        n_sub = len(attn_subs)
        cs = (n_cols // g_col) // n_sub
        n_cols_sub = g_col * cs
        plans, attn_c, gcn_c = [], [], []
        for s_ in range(n_sub):
            if use_col:
                _, cs_need = need(attn_subs[s_], gcn_subs[s_] if gcn_subs is not None else None, n_cols_sub)
                p = SparseRowPlan(grid.col_group, cs, cs_need, name=f"col{s_}")
                col_remap, cols_eff = p.remap.long(), p.need_local.long()
            else:
                p, col_remap = None, torch.arange(n_cols_sub, device=dev)
                cols_eff = col_remap
            plans.append(p)
            attn_c.append(attn_subs[s_].compacted(rows_eff, col_remap))
            gcn_c.append(GCNBlockView(gcn_subs[s_], rows_eff, n_rows_c, col_remap, cols_eff.numel(), cols_eff)
                         if gcn_subs is not None else None)
            if p is not None:
                p.drop_remap()
            del col_remap
        out["cols"], out["attn"], out["gcn"] = plans, attn_c, gcn_c
    for p in (plan_row, plan_col):
        p.drop_remap()
    return out, info


class DistExphormerLayerFused(nn.Module):
    def __init__(self, grid, gcn_block, attn_block, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                 comm_dtype=None, rs_dtype=None, with_gcn=True, gather_proj=False, kernel_dtype=None, sparse=None):
        super().__init__()
        self.grid = grid
        self.dim_h, self.num_heads, self.d_head = dim_h, num_heads, dim_h // num_heads
        self.dropout, self.comm_dtype, self.rs_dtype = dropout, comm_dtype, rs_dtype
        # sparse: build_sparse_comm's dict — the row / column plans (None = dense collective); the blocks given to this
        # layer are then the compact tiles (rows = the row plan's rows, columns = the column plan's rows)
        self.sp_row = sparse["row"] if sparse is not None else None
        self.sp_col = sparse["col"] if sparse is not None else None
        # kernel_dtype (e.g. bfloat16): Q / K / V are stored and read by the attention kernels in this dtype (fp32
        # accumulation inside): halves the per-edge bytes of the kernels and the saved K / V / Q of every layer
        self.kernel_dtype = kernel_dtype
        # gather_proj: the owner projects its chunk once and the PROJECTIONS are gathered (Q over the row group, width
        # d; [K | V | GCN message] over the column group, width 3d) instead of x (width d) followed by redundant
        # projections of the gathered blocks on every rank: P_d x fewer projection FLOPs for 3x the column-group bytes.
        self.gather_proj = gather_proj
        self.gcn_block = gcn_block
        self.with_gcn = with_gcn                                              # False: attention-only layer (no GCN parameters)
        if with_gcn:
            # same parameter names as DistExphormerLayer / ExphormerLayerRef
            self.gcn = nn.Module()
            self.gcn.lin = nn.Linear(dim_h, dim_h, bias=False)
            self.gcn.bias = nn.Parameter(torch.zeros(dim_h))
            self.bn_local = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        self.q = nn.Linear(dim_h, dim_h, bias=False)
        self.k = nn.Linear(dim_h, dim_h, bias=False)
        self.v = nn.Linear(dim_h, dim_h, bias=False)
        self.e = nn.Linear(dim_h, dim_h, bias=False)
        self.attn = DistAttentionPartial(grid, attn_block, num_heads, self.d_head, backend=backend,
                                         score="gated", norm="clampexp")
        self.bn_attn = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        self.ff1 = nn.Linear(dim_h, 2 * dim_h)
        self.ff2 = nn.Linear(2 * dim_h, dim_h)
        self.bn_ff = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)

    def forward(self, x, edge_type_emb):
        g = self.grid
        n, h, d = x.shape[0], self.num_heads, self.d_head
        cdt = self.comm_dtype or x.dtype
        kdt = self.kernel_dtype or x.dtype
        t = _timers()
        gate = self.e(edge_type_emb).view(-1, h, d)
        agg = None
        if self.gather_proj:
            hd = h * d
            t.start("qkv proj")                                              # owned rows only
            q_loc = self.q(x)
            kvh_loc = torch.cat([self.k(x), self.v(x)] + ([self.gcn.lin(x)] if self.with_gcn else []), dim=1)
            t.stop("qkv proj")
            t.start("gather x")                                              # (gathers of the projections)
            q = _gather_rows(q_loc.to(cdt), g.row_group, self.sp_row).to(x.dtype).view(-1, h, d)   # [P_o*chunk, h, d]
            kvh = _gather_rows(kvh_loc.to(cdt), g.col_group, self.sp_col).to(x.dtype)              # [P_d*chunk, 2-3 hd]
            t.stop("gather x")
            k = kvh[:, :hd].contiguous().view(-1, h, d)
            v = kvh[:, hd:2 * hd].contiguous().view(-1, h, d)
            if self.with_gcn:
                t.start("gcn spmm")
                a, at = self.gcn_block.matrices(x.dtype)
                agg = spmm(kvh[:, 2 * hd:].contiguous(), a, at)                # [P_o*chunk, dim]
                t.stop("gcn spmm")
        else:
            t.start("gather x")
            x_col = _gather_rows(x.to(cdt), g.col_group, self.sp_col).to(x.dtype)    # [P_d*chunk, dim]  column block
            x_row = _gather_rows(x.to(cdt), g.row_group, self.sp_row).to(x.dtype)    # [P_o*chunk, dim]  row block
            t.stop("gather x")
            # GCN branch: aggregate (X W) over the local block -> row-block partial
            if self.with_gcn:
                t.start("gcn spmm")
                a, at = self.gcn_block.matrices(x.dtype)
                agg = spmm(self.gcn.lin(x_col), a, at)                           # [P_o*chunk, dim]
                t.stop("gcn spmm")
            # attention branch on the gathered blocks -> unnormalised partials
            t.start("qkv proj")
            q = self.q(x_row).to(kdt).view(-1, h, d)
            k = self.k(x_col).to(kdt).view(-1, h, d)
            v = self.v(x_col).to(kdt).view(-1, h, d)
            t.stop("qkv proj")
        t.start("attention partial")
        acc, l, m = self.attn(q, k, v, gate)
        t.stop("attention partial")
        t.start("reduce-scatter")
        o, agg_o = combine_partials(acc, l, m, g, "clampexp", 1e-6, extra=agg, rs_dtype=self.rs_dtype, plan=self.sp_row)   # one reduce-scatter
        t.stop("reduce-scatter")
        h_attn = F.dropout(o.reshape(n, h * d).to(x.dtype), self.dropout, training=self.training)
        if not self.with_gcn:
            hh = self.bn_attn(x + h_attn)
        else:
            h_local = agg_o + self.gcn.bias
            h_local = F.dropout(h_local, self.dropout, training=self.training)
            if BN_PAIR and _is_bn(self.bn_local):
                h_local, h_attn = dist_batchnorm_pair(self.bn_local, x + h_local, self.bn_attn, x + h_attn)   # one all-reduce
            else:
                h_local, h_attn = self.bn_local(x + h_local), self.bn_attn(x + h_attn)
            hh = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(hh + ff)


# =============================================================================================
# Head-axis layer (Grid3D = 2D block grid x DeepSpeed-Ulysses head axis), the Exphormer layer with or without the GCN:
#   the OWNER projects its chunk once (Q, K, V, all heads, and the GCN message X W_g); rows <-> heads with an all-to-all
#   inside the head group — the GCN message is split along its d feature columns exactly like the heads (A (X W) is
#   column-separable), packed with K and V into ONE all-to-all and ONE all-gather over the column group; Q is gathered
#   over the row group; the block kernel runs on H / P_i heads and the GCN SpMM on the d / P_i feature slice of the
#   same block; the attention partials and the GCN partial ride in ONE reduce-scatter over the row group; the
#   normalised attention output and the GCN slice go back rows-wise in ONE all-to-all.  Per layer and direction:
#   2 all-to-alls, 2 all-gathers, 1 reduce-scatter (the attention-only operator of bench_dist_attention uses 4 + 3 + 1).
#   Same parameters / semantics as DistExphormerLayer / ExphormerLayerRef (with or without the GCN branch).
#   Bytes per rank: a2a (3 or 2) x O d (1 - 1/P_i), gathers ((3 or 2) N/P_o + N/P_d) d / P_i, reduce-scatter
#   N/P_d (2 or 1) d / P_i, versus the fused 2D layer's (N/P_o + N/P_d) d and N/P_d (2 d + H).
# =============================================================================================
# GT3D_HEAD_MERGED=0: the attention-only head layer uses three separate all-to-alls / gathers for Q, K, V (the operator path
# of this morning) instead of the merged [K | V] tensor — an A/B switch for the layer's communication pattern.
HEAD_MERGED = os.environ.get("GT3D_HEAD_MERGED", "1") == "1"


def _head_attention_separate(layer, q, k, v, gate):
    from gt3d.dist.grid3d import rows_to_heads, heads_to_rows
    g, hg, ds = layer.grid, layer.hg, layer.ds
    cdt = layer.comm_dtype

    def to_block(t_, group):
        wire = t_.to(cdt) if cdt is not None else t_
        return _all_gather_rows(rows_to_heads(wire, g.head_group, g.p_heads), group).to(t_.dtype)
    q_blk = to_block(q.contiguous(), g.row_group)
    k_blk = to_block(k.contiguous(), g.col_group)
    v_blk = to_block(v.contiguous(), g.col_group)
    gate_h = gate[:, g.h * hg:(g.h + 1) * hg].contiguous()
    acc, l, m = layer.attn(q_blk, k_blk, v_blk, gate_h)
    o_h, _ = combine_partials(acc, l, m, g, "clampexp", 1e-6, extra=None, rs_dtype=layer.rs_dtype)
    wire = o_h.to(cdt) if cdt is not None else o_h
    return heads_to_rows(wire, g.head_group, g.p_heads).to(q.dtype)


class DistExphormerLayerHead(nn.Module):
    def __init__(self, grid, attn_block, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                 comm_dtype=None, rs_dtype=None, kernel_dtype=None, gcn_block=None, with_gcn=False, sparse=None):
        super().__init__()
        from gt3d.dist.sparse_attention import DistAttentionPartial
        self.grid = grid
        self.sp_row = sparse["row"] if sparse is not None else None        # see DistExphormerLayerFused
        self.sp_col = sparse["col"] if sparse is not None else None
        assert num_heads % grid.p_heads == 0 and dim_h % grid.p_heads == 0, (num_heads, dim_h, grid.p_heads)
        self.dim_h, self.num_heads, self.d_head = dim_h, num_heads, dim_h // num_heads
        self.hg, self.ds = num_heads // grid.p_heads, dim_h // grid.p_heads        # heads / feature columns per head group
        self.dropout, self.comm_dtype, self.rs_dtype, self.kernel_dtype = dropout, comm_dtype, rs_dtype, kernel_dtype
        self.with_gcn = with_gcn
        self.gcn_block = gcn_block
        if with_gcn:
            assert gcn_block is not None
            self.gcn = nn.Module()
            self.gcn.lin = nn.Linear(dim_h, dim_h, bias=False)
            self.gcn.bias = nn.Parameter(torch.zeros(dim_h))
            self.bn_local = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        self.q = nn.Linear(dim_h, dim_h, bias=False)
        self.k = nn.Linear(dim_h, dim_h, bias=False)
        self.v = nn.Linear(dim_h, dim_h, bias=False)
        self.e = nn.Linear(dim_h, dim_h, bias=False)
        self.attn = DistAttentionPartial(grid, attn_block, self.hg, self.d_head, backend=backend, score="gated", norm="clampexp")
        self.bn_attn = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)
        self.ff1 = nn.Linear(dim_h, 2 * dim_h)
        self.ff2 = nn.Linear(2 * dim_h, dim_h)
        self.bn_ff = make_norm(NORM_KIND[0], grid, dim_h, bn_eps, bn_momentum)

    def forward(self, x, edge_type_emb):
        from gt3d.dist.grid3d import rows_to_heads, heads_to_rows
        g = self.grid
        n, h, d, p_i, hg, ds = x.shape[0], self.num_heads, self.d_head, g.p_heads, self.hg, self.ds
        cdt = self.comm_dtype or x.dtype
        kdt = self.kernel_dtype or x.dtype
        t = _timers()
        gate = self.e(edge_type_emb).view(-1, h, d)
        if not self.with_gcn and not HEAD_MERGED:                                # A/B: the separate Q / K / V path of the operator
            assert self.sp_row is None and self.sp_col is None, "GT3D_HEAD_MERGED=0 has no sparse-communication path"
            t.start("qkv proj")
            q = self.q(x).to(kdt).view(n, h, d); k = self.k(x).to(kdt).view(n, h, d); v = self.v(x).to(kdt).view(n, h, d)
            t.stop("qkv proj")
            t.start("head attention")
            o = _head_attention_separate(self, q, k, v, gate)
            t.stop("head attention")
            h_attn = F.dropout(o.reshape(n, h * d).to(x.dtype), self.dropout, training=self.training)
            hh = self.bn_attn(x + h_attn)
            ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
            ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
            return self.bn_ff(hh + ff)
        t.start("qkv proj")                                                      # owned rows, all heads
        q = self.q(x).to(kdt).view(n, p_i, ds)
        # ONE projection [n, (2 | 3) d] whose columns are already ordered per head group j as [K_j | V_j | G_j]: the weight
        # rows of k / v / gcn.lin are interleaved per slice (a d x d-sized copy of the parameters, their gradients flow back
        # through the view), so the all-to-all input [n, P_i, (2 | 3) ds] is a free view — no torch.cat of activations
        ws = [self.k.weight.view(p_i, ds, self.dim_h), self.v.weight.view(p_i, ds, self.dim_h)]
        if self.with_gcn:
            ws.append(self.gcn.lin.weight.view(p_i, ds, self.dim_h))
        w_kvg = torch.cat(ws, dim=1).view(-1, self.dim_h)                        # [(2 | 3) d, d]
        kvg = F.linear(x, w_kvg).view(n, p_i, -1)                                # [n, P_i, (2 | 3) ds] (model dtype)
        t.stop("qkv proj")
        t.start("head: a2a + gather q/k/v")
        q_blk = _gather_rows(rows_to_heads(q.to(cdt), g.head_group, p_i), g.row_group, self.sp_row).to(kdt).view(-1, hg, d)
        kvg_blk = _gather_rows(rows_to_heads(kvg.to(cdt), g.head_group, p_i), g.col_group, self.sp_col).view(-1, kvg.shape[2])
        t.stop("head: a2a + gather q/k/v")
        parts = kvg_blk.split(ds, dim=1)                                         # split: one cat in the backward, no zero-fills
        k_blk = parts[0].to(kdt).contiguous().view(-1, hg, d)
        v_blk = parts[1].to(kdt).contiguous().view(-1, hg, d)
        agg = None
        if self.with_gcn:
            t.start("gcn spmm")
            a, at = self.gcn_block.matrices(x.dtype)
            agg = spmm(parts[2].to(x.dtype).contiguous(), a, at)                 # [rows, ds]: the feature slice's partial
            t.stop("gcn spmm")
        gate_h = gate[:, g.h * hg:(g.h + 1) * hg].contiguous()
        t.start("attention partial")
        acc, l, m = self.attn(q_blk, k_blk, v_blk, gate_h)
        t.stop("attention partial")
        t.start("reduce-scatter")
        o_h, agg_o = combine_partials(acc, l, m, g, "clampexp", 1e-6, extra=agg, rs_dtype=self.rs_dtype, plan=self.sp_row)   # one reduce-scatter
        t.stop("reduce-scatter")
        t.start("head: a2a back")
        back = o_h.reshape(-1, ds) if agg_o is None else torch.cat([o_h.reshape(-1, ds), agg_o], dim=1)
        back = heads_to_rows(back.to(cdt).view(-1, 1, back.shape[1]), g.head_group, p_i).to(x.dtype)   # [n, P_i, (1 | 2) ds]
        t.stop("head: a2a back")
        o = back[:, :, :ds].reshape(n, h * d)
        h_attn = F.dropout(o, self.dropout, training=self.training)
        if not self.with_gcn:
            hh = self.bn_attn(x + h_attn)
        else:
            h_local = back[:, :, ds:].reshape(n, self.dim_h) + self.gcn.bias
            h_local = F.dropout(h_local, self.dropout, training=self.training)
            if BN_PAIR and _is_bn(self.bn_local):
                h_local, h_attn = dist_batchnorm_pair(self.bn_local, x + h_local, self.bn_attn, x + h_attn)   # one all-reduce
            else:
                h_local, h_attn = self.bn_local(x + h_local), self.bn_attn(x + h_attn)
            hh = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(hh + ff)


class DistExphormerLayerPipelined(DistExphormerLayerFused):
    """Fused layer with the column-block input gathered in n_sub sub-blocks; the gather of sub-block s+1
    (asynchronous) overlaps the GCN projection/SpMM, the K/V projections and the attention partial of
    sub-block s.  The row-block gather (Q) is started first and overlaps the first sub-block's compute.
    Same mathematics as DistExphormerLayerFused (partials add up; clamp-exp needs no rescaling)."""

    def __init__(self, grid, gcn_blocks, attn_blocks, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                 comm_dtype=None, rs_dtype=None, kernel_dtype=None, with_gcn=True):
        # gcn_blocks / attn_blocks: lists (one per column sub-block) built by the model
        super().__init__(grid, gcn_blocks[0], attn_blocks[0], dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                         comm_dtype=comm_dtype, rs_dtype=rs_dtype, kernel_dtype=kernel_dtype, with_gcn=with_gcn)
        self.gcn_blocks = gcn_blocks
        self.attns = nn.ModuleList([DistAttentionPartial(grid, b, num_heads, self.d_head, backend=backend,
                                                         score="gated", norm="clampexp") for b in attn_blocks])
        self.n_sub = len(gcn_blocks)
        self.cs = grid.chunk // self.n_sub

    def forward(self, x, edge_type_emb):
        g = self.grid
        n, h, d = x.shape[0], self.num_heads, self.d_head
        cdt = self.comm_dtype or x.dtype
        xc = x.to(cdt)
        x_row_buf = all_gather_rows_start(xc, g.row_group)                        # Q side, overlaps sub-block 0
        pending = all_gather_rows_start(xc[0:self.cs].contiguous(), g.col_group)   # sub-block 0
        gate = self.e(edge_type_emb).view(-1, h, d)
        q = None
        acc_tot = l_tot = agg_tot = None
        for s in range(self.n_sub):
            xs = wait_gather(pending).to(x.dtype)
            if s + 1 < self.n_sub:                                                # prefetch the next sub-block
                pending = all_gather_rows_start(xc[(s + 1) * self.cs:(s + 2) * self.cs].contiguous(), g.col_group)
            a, at = self.gcn_blocks[s].matrices(x.dtype)
            agg_s = spmm(self.gcn.lin(xs), a, at)
            ks = self.k(xs).view(-1, h, d)
            vs = self.v(xs).view(-1, h, d)
            if q is None:
                q = self.q(wait_gather(x_row_buf).to(x.dtype)).view(-1, h, d)
            acc_s, l_s, _ = self.attns[s](q, ks, vs, gate)
            acc_tot = acc_s if acc_tot is None else acc_tot + acc_s
            l_tot = l_s if l_tot is None else l_tot + l_s
            agg_tot = agg_s if agg_tot is None else agg_tot + agg_s
        m_dummy = torch.zeros_like(l_tot)
        o, agg_o = combine_partials(acc_tot, l_tot, m_dummy, g, "clampexp", 1e-6, extra=agg_tot, rs_dtype=self.rs_dtype)
        h_local = agg_o + self.gcn.bias
        h_local = F.dropout(h_local, self.dropout, training=self.training)
        h_attn = F.dropout(o.reshape(n, h * d).to(x.dtype), self.dropout, training=self.training)
        if BN_PAIR and _is_bn(self.bn_local):
            h_local, h_attn = dist_batchnorm_pair(self.bn_local, x + h_local, self.bn_attn, x + h_attn)   # one all-reduce
        else:
            h_local, h_attn = self.bn_local(x + h_local), self.bn_attn(x + h_attn)
        hh = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(hh + ff)


# =============================================================================================
# Streamed layer: the column block is never materialised.  Forward gathers the column block in n_sub
# sub-blocks, projects K/V/GCN on each, accumulates the unnormalised partials and DISCARDS the sub-block;
# backward re-gathers and re-projects each sub-block (flash-attention-style recomputation) and
# reduce-scatters its input gradient.  Saved for backward: the owned input only.  Per-rank memory is
# O(N/P_d) (Q, partials) + O(N/(P_o n_sub)) (one sub-block) instead of O(N/P_o) for K, V, x_col;
# communication is the same as the fused layer's (one gather per sub-block and direction, one
# reduce-scatter of dx per sub-block, plus the row-block gather / reduce-scatter for Q).
# =============================================================================================
class _StreamedAttnGCN(torch.autograd.Function):
    """Returns the OWNED rows of the reduce-scattered payload (acc | l | gcn partial), [chunk, h*d + h + dim].
    Row-block-sized tensors alive at any time: the bf16 row block, Q (fp32) and, in backward, dQ, dacc, dagg;
    the projections from the bf16 row block are chunked so no fp32 copy of it exists."""

    # Every collective takes an optional SparseRowPlan (layer.sp_row for the row group, layer.sp_cols[s] for column
    # sub-block s; None = dense): with a plan the gathered buffers hold only the plan's rows and the tiles given to the
    # layer are the compact ones (build_sparse_comm).
    @staticmethod
    def _gather(x, group, plan=None):
        return raw_all_gather_rows(x, group) if plan is None else plan.gather(x)

    # ---- communication / compute overlap (layer.overlap, `--stream_overlap`): the gather of column sub-block s+1 is
    # started before sub-block s is computed, and in backward the reduce-scatter of dx_s is finished only after
    # sub-block s+1 has been computed.  One extra sub-block buffer in flight.
    @staticmethod
    def _gather_start(x, group, plan=None):
        if plan is not None:
            return plan.gather(x, async_op=True)                                   # (work, fin)
        return raw_all_gather_rows(x.contiguous(), group, async_op=True)          # (work, fin)

    @staticmethod
    def _gather_wait(pend):
        work, fin = pend
        if work is not None:
            from gt3d.dist.sparse_comm import wait_work
            wait_work(work)
        return fin()

    @staticmethod
    def _rs_start(x, group, s, plan=None):
        """asynchronous reduce-scatter over the rows (torch.distributed directly: the Plexus wrapper has no async_op);
        returns (work, finalize, s)."""
        if plan is not None:
            work, fin = plan.reduce_scatter(x, async_op=True)
            return work, fin, s
        ws = dist.get_world_size(group)
        if ws == 1:
            return None, (lambda: x), s
        x = x.contiguous()
        out = x.new_empty(x.shape[0] // ws, *x.shape[1:])
        work = dist.reduce_scatter_tensor(out, x, group=group, async_op=True)
        return work, (lambda _x=x: out), s                                        # the default arg keeps x alive until done

    @staticmethod
    def _rs_finish(pend, dx, cs, dtype):
        work, fin, s = pend
        if work is not None:
            from gt3d.dist.sparse_comm import wait_work
            wait_work(work)
        dx[s * cs:(s + 1) * cs] += fin().to(dtype)

    @staticmethod
    def _reduce_scatter(x, group, plan=None):
        return raw_reduce_scatter_rows(x, group) if plan is None else plan.reduce_scatter(x)

    @staticmethod
    def _kernels(layer):
        if layer.backend == "triton":
            from gt3d.attention.triton_kernel import attn_fwd_partial, attn_bwd_partial
            return attn_fwd_partial, attn_bwd_partial, lambda t: t, lambda t: (t if t is None else t)
        from gt3d.attention.backend_torch import torch_attn_fwd_partial, torch_attn_bwd_partial
        return torch_attn_fwd_partial, torch_attn_bwd_partial, lambda t: t.long(), lambda t: (None if t is None else t.long())

    @staticmethod
    def _accum_kernels(layer):
        """(fwd_accum, bwd_accum) of the compact-accumulate path (triton backend, layer.compact): the attention
        partials of a sub-block are added in place to the packed buffer / dq by kernels that tile only the rows
        owning an edge in the sub-block (DCSR), instead of a full-row-block partial plus a full-row-block add."""
        if layer.backend != "triton" or not getattr(layer, "compact", True):
            return None, None
        from gt3d.attention.triton_kernel import attn_fwd_partial_accum, attn_bwd_partial_accum
        return attn_fwd_partial_accum, attn_bwd_partial_accum

    @staticmethod
    def _rows_per_chunk(dim):
        return max(1, (1 << 30) // (dim * 4))                     # ~1 GB fp32 per chunk

    @staticmethod
    def _proj(x_lp, w, dtype):
        """(x_lp.to(dtype) @ w.T) computed in row chunks: x_lp is the low-precision gathered block."""
        out = torch.empty(x_lp.shape[0], w.shape[0], dtype=dtype, device=x_lp.device)
        cr = _StreamedAttnGCN._rows_per_chunk(x_lp.shape[1])
        for i in range(0, x_lp.shape[0], cr):
            out[i:i + cr] = x_lp[i:i + cr].to(dtype) @ w.t()
        return out

    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, gate, layer):
        L, g = layer, layer.grid
        h, d, cs = L.num_heads, L.d_head, L.cs
        fwd, _, ci, ei = _StreamedAttnGCN._kernels(L)
        fwd_acc, _ = _StreamedAttnGCN._accum_kernels(L)
        cmp = getattr(L, "compact", True)                   # DCSR GCN SpMM (any backend); attention kernels: triton only
        gate_c = gate.to(torch.float32 if L.backend == "triton" else x.dtype)
        cdt = L.comm_dtype or x.dtype
        t = _timers()
        sp_row, sp_cols = getattr(L, "sp_row", None), getattr(L, "sp_cols", None)
        spc = (lambda s_: sp_cols[s_]) if sp_cols is not None else (lambda s_: None)     # noqa: E731
        with torch.no_grad():
            xc = x.to(cdt).contiguous()
            ov = getattr(L, "overlap", False)
            pend = _StreamedAttnGCN._gather_start(xc[0:cs], g.col_group, spc(0)) if ov else None   # sub-block 0 overlaps the Q side
            t.start("stream fwd: gather x_row + q")
            x_row = _StreamedAttnGCN._gather(xc, g.row_group, sp_row)              # [rows, dim] comm dtype
            kdt = L.kernel_dtype or x.dtype
            q = _StreamedAttnGCN._proj(x_row, wq, x.dtype).to(kdt).view(-1, h, d)
            del x_row
            t.stop("stream fwd: gather x_row + q")
            rows, dim, hd = q.shape[0], x.shape[1], h * d
            packed = None                                       # [rows, h*d + h + dim] = (acc, l, gcn partial): the reduce-scatter payload
            for s in range(L.n_sub):
                t.start("stream fwd: gather sub-block")
                if ov:                                          # wait for sub-block s; the gather of s+1 overlaps the compute below
                    xs = _StreamedAttnGCN._gather_wait(pend).to(x.dtype)
                    pend = _StreamedAttnGCN._gather_start(xc[(s + 1) * cs:(s + 2) * cs], g.col_group, spc(s + 1)) if s + 1 < L.n_sub else None
                else:
                    xs = _StreamedAttnGCN._gather(xc[s * cs:(s + 1) * cs], g.col_group, spc(s)).to(x.dtype)
                t.stop("stream fwd: gather sub-block")
                t.start("stream fwd: gcn + kv proj")
                if cmp:                                         # DCSR SpMM: only the rows with an edge in the sub-block
                    rows_c, a_c = L.gcn_blocks[s].matrices_compact(x.dtype)
                    agg_s = torch.sparse.mm(a_c, xs @ wg.t()) if rows_c.numel() > 0 else None
                else:
                    a, _ = L.gcn_blocks[s].matrices(x.dtype)
                    agg_s = torch.sparse.mm(a, xs @ wg.t())
                ks = (xs @ wk.t()).to(kdt).view(-1, h, d)
                vs = (xs @ wv.t()).to(kdt).view(-1, h, d)
                del xs
                t.stop("stream fwd: gcn + kv proj")
                blk = L.attns[s]
                t.start("stream fwd: attention")
                if fwd_acc is not None:                         # compact-accumulate: partials added in place
                    if packed is None:
                        packed = torch.zeros(rows, hd + h + dim, dtype=torch.float32, device=q.device)
                    fwd_acc(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None,
                            packed[:, :hd], packed[:, hd:hd + h],
                            score=L.attn_score, norm="clampexp", clamp=L.attn_clamp, scale=L.attn_scale)
                else:
                    acc_s, l_s, _ = fwd(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None,
                                        score=L.attn_score, norm="clampexp", clamp=L.attn_clamp, scale=L.attn_scale)
                    if packed is None:
                        packed = acc_s.new_zeros(rows, hd + h + dim)
                    packed[:, :hd] += acc_s.view(rows, hd)
                    packed[:, hd:hd + h] += l_s
                    del acc_s, l_s
                t.stop("stream fwd: attention")
                del ks, vs
                if cmp:
                    if agg_s is not None:
                        packed[:, hd + h:].index_add_(0, rows_c, agg_s.to(packed.dtype))
                else:
                    packed[:, hd + h:] += agg_s.to(packed.dtype)
                del agg_s
            del q
            ctx.packed_dtype = packed.dtype
            if L.rs_dtype is not None and L.rs_dtype != packed.dtype:
                packed = packed.to(L.rs_dtype)
            t.start("stream fwd: reduce-scatter")
            packed_o = _StreamedAttnGCN._reduce_scatter(packed, g.row_group, sp_row)     # owned rows
            t.stop("stream fwd: reduce-scatter")
            del packed
        ctx.save_for_backward(x, wq, wk, wv, wg, gate)
        ctx.layer = L
        ctx.gate_c_dtype = gate_c.dtype
        return packed_o

    @staticmethod
    def backward(ctx, dpo):
        x, wq, wk, wv, wg, gate = ctx.saved_tensors
        L, g = ctx.layer, ctx.layer.grid
        h, d, cs = L.num_heads, L.d_head, L.cs
        _, bwd, ci, ei = _StreamedAttnGCN._kernels(L)
        _, bwd_acc = _StreamedAttnGCN._accum_kernels(L)
        gate_c = gate.to(ctx.gate_c_dtype)
        cdt = L.comm_dtype or x.dtype
        pdt = ctx.packed_dtype
        t = _timers()
        sp_row, sp_cols = getattr(L, "sp_row", None), getattr(L, "sp_cols", None)
        spc = (lambda s_: sp_cols[s_]) if sp_cols is not None else (lambda s_: None)     # noqa: E731
        t.start("stream bwd")
        with torch.no_grad():
            hd = h * d
            dpo = dpo.contiguous()
            # gradients of the row-block partials: all-gather the three pieces separately (no packed copy)
            dacc = _StreamedAttnGCN._gather(dpo[:, :hd].contiguous(), g.row_group, sp_row).to(pdt)
            dl = _StreamedAttnGCN._gather(dpo[:, hd:hd + h].contiguous(), g.row_group, sp_row).to(pdt)
            dagg = _StreamedAttnGCN._gather(dpo[:, hd + h:].contiguous(), g.row_group, sp_row).to(x.dtype)
            del dpo
            rows = dacc.shape[0]
            dacc = dacc.view(rows, h, d)
            xc = x.to(cdt).contiguous()
            ov = getattr(L, "overlap", False)
            pend = _StreamedAttnGCN._gather_start(xc[0:cs], g.col_group, spc(0)) if ov else None   # overlaps the Q side
            rs_pend = None                                                         # previous sub-block's dx reduce-scatter
            x_row = _StreamedAttnGCN._gather(xc, g.row_group, sp_row)              # comm dtype
            kdt = L.kernel_dtype or x.dtype
            q = _StreamedAttnGCN._proj(x_row, wq, x.dtype).to(kdt).view(-1, h, d)
            if kdt != x.dtype and L.backend == "triton":
                dacc = dacc.to(kdt)                                                # read per edge in the column pass
            m0 = torch.zeros_like(dl)                                              # clamp-exp partials have m = 0
            # dq accumulates over the sub-blocks in fp32 (the compact-accumulate kernels add into it); the fp64 reference
            # path (torch backend) keeps fp64, otherwise its gradients carry a 5e-8 fp32 rounding error
            dq = torch.zeros(q.shape, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32, device=q.device)
            dwk, dwv, dwg = torch.zeros_like(wk), torch.zeros_like(wv), torch.zeros_like(wg)
            dgate = None
            dx = torch.zeros_like(x)
            for s in range(L.n_sub):
                if ov:
                    xs = _StreamedAttnGCN._gather_wait(pend).to(x.dtype)
                    pend = _StreamedAttnGCN._gather_start(xc[(s + 1) * cs:(s + 2) * cs], g.col_group, spc(s + 1)) if s + 1 < L.n_sub else None
                else:
                    xs = _StreamedAttnGCN._gather(xc[s * cs:(s + 1) * cs], g.col_group, spc(s)).to(x.dtype)
                _, at = L.gcn_blocks[s].matrices(x.dtype)
                ks = (xs @ wk.t()).to(kdt).view(-1, h, d)
                vs = (xs @ wv.t()).to(kdt).view(-1, h, d)
                blk = L.attns[s]
                if bwd_acc is not None:                         # compact-accumulate: dq added in place
                    dks, dvs, dgate_s, _ = bwd_acc(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None, m0, dacc, dl, dq,
                                                   score=L.attn_score, norm="clampexp", clamp=L.attn_clamp, scale=L.attn_scale)
                else:
                    dq_s, dks, dvs, dgate_s, _ = bwd(q, ks, vs, blk.crow, ci(blk.col), ei(blk.etype), gate_c, None, m0, dacc, dl,
                                                     score=L.attn_score, norm="clampexp", clamp=L.attn_clamp, scale=L.attn_scale)
                    dq += dq_s.to(q.dtype)
                    del dq_s
                del ks, vs
                if dgate_s is not None:
                    dgate = dgate_s if dgate is None else dgate + dgate_s
                dks, dvs = dks.to(x.dtype).view(-1, hd), dvs.to(x.dtype).view(-1, hd)
                dp_s = torch.sparse.mm(at, dagg)                                   # grad of the GCN projection xs @ wg^T
                dwk += dks.t() @ xs; dwv += dvs.t() @ xs; dwg += dp_s.t() @ xs
                del xs
                dxs = dks @ wk
                dxs += dvs @ wv
                dxs += dp_s @ wg                                                    # [cols_sub, dim]
                del dks, dvs, dp_s
                if ov:                                          # the reduce-scatter of dx_s overlaps the next sub-block
                    if rs_pend is not None:
                        _StreamedAttnGCN._rs_finish(rs_pend, dx, cs, x.dtype)
                    rs_pend = _StreamedAttnGCN._rs_start(dxs.to(cdt), g.col_group, s, spc(s))
                else:
                    dx[s * cs:(s + 1) * cs] += _StreamedAttnGCN._reduce_scatter(dxs.to(cdt), g.col_group, spc(s)).to(x.dtype)
                del dxs
            if rs_pend is not None:
                _StreamedAttnGCN._rs_finish(rs_pend, dx, cs, x.dtype)
            del dacc, dl, dagg, q
            dq2 = dq.view(rows, hd)
            cr = _StreamedAttnGCN._rows_per_chunk(x.shape[1])
            dwq = torch.zeros_like(wq)
            dxr = torch.empty(rows, x.shape[1], dtype=cdt, device=x.device)
            for i in range(0, rows, cr):
                xr = x_row[i:i + cr].to(x.dtype)
                dqi = dq2[i:i + cr].to(x.dtype)                 # dq accumulates in fp32 (a no-op cast except for the fp64 torch backend)
                dwq += dqi.t() @ xr
                dxr[i:i + cr] = (dqi @ wq).to(cdt)
            del x_row, dq, dq2
            dx += _StreamedAttnGCN._reduce_scatter(dxr, g.row_group, sp_row).to(x.dtype)
        t.stop("stream bwd")
        return dx, dwq, dwk, dwv, dwg, (dgate.to(gate.dtype) if dgate is not None else None), None


class DistExphormerLayerStreamed(DistExphormerLayerPipelined):
    """Fused-communication layer with the column block streamed in n_sub sub-blocks and recomputed in
    backward (see _StreamedAttnGCN).  With tail_checkpoint the owned-row part (dropout, BatchNorm, FFN)
    is checkpointed too, so the layer keeps only its input and the two reduce-scatter outputs between
    forward and backward and the model needs no layer-level checkpoint (which would re-gather everything)."""

    self_checkpointing = True

    def __init__(self, grid, gcn_blocks, attn_blocks, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                 comm_dtype=None, rs_dtype=None, tail_checkpoint=False, kernel_dtype=None, with_gcn=True):
        super().__init__(grid, gcn_blocks, attn_blocks, dim_h, num_heads, dropout, bn_eps, bn_momentum, backend,
                         comm_dtype=comm_dtype, rs_dtype=rs_dtype, kernel_dtype=kernel_dtype, with_gcn=with_gcn)
        self.backend, self.tail_checkpoint = backend, tail_checkpoint
        a0 = self.attns[0]
        self.attn_score, self.attn_clamp, self.attn_scale = a0.score, a0.clamp, a0.scale

    def _tail(self, x, packed_o):
        n, h, d = x.shape[0], self.num_heads, self.d_head
        hd = h * d
        l_o = packed_o[:, hd:hd + h]
        o = (packed_o[:, :hd].view(n, h, d) / (l_o + 1e-6).unsqueeze(-1)).to(x.dtype)
        if not self.with_gcn:                                                    # attention-only layer (ring)
            h_attn = F.dropout(o.reshape(n, -1).to(x.dtype), self.dropout, training=self.training)
            hh = self.bn_attn(x + h_attn)
            ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
            ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
            return self.bn_ff(hh + ff)
        agg_o = packed_o[:, hd + h:].to(x.dtype)
        h_local = agg_o + self.gcn.bias
        h_local = F.dropout(h_local, self.dropout, training=self.training)
        h_attn = F.dropout(o.reshape(n, -1).to(x.dtype), self.dropout, training=self.training)
        if BN_PAIR and _is_bn(self.bn_local):
            h_local, h_attn = dist_batchnorm_pair(self.bn_local, x + h_local, self.bn_attn, x + h_attn)
        else:
            h_local, h_attn = self.bn_local(x + h_local), self.bn_attn(x + h_attn)
        hh = h_local + h_attn
        ff = F.dropout(F.relu(self.ff1(hh)), self.dropout, training=self.training)
        ff = F.dropout(self.ff2(ff), self.dropout, training=self.training)
        return self.bn_ff(hh + ff)

    def forward(self, x, edge_type_emb):
        h, d = self.num_heads, self.d_head
        gate = self.e(edge_type_emb).view(-1, h, d)
        packed_o = _StreamedAttnGCN.apply(x, self.q.weight, self.k.weight, self.v.weight, self.gcn.lin.weight, gate, self)
        if self.tail_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._tail, x, packed_o, use_reentrant=False)
        return self._tail(x, packed_o)
