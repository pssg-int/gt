"""Sparse (all-to-all-v, static index sets) vs dense (NCCL all-gather / reduce-scatter) row collectives on the real
tiles of a partition: the primitive cost of gt3d/dist/sparse_comm.py and the bytes it saves, per group layout.

For the grid (P_d, P_o) and the rank's attention tile (graph edges + self loops + expander edges), it builds the row /
column plans with every collective forced sparse and times, in the comm dtype (bf16) for hidden sizes --hidden:
  col gather  : x of the column block   [P_d chunk, dim]   dense all-gather vs plan.gather
  col RS      : dx of the column block  [P_d chunk, dim]   dense reduce-scatter vs plan.reduce_scatter
  row gather  : x of the row block      [P_o chunk, dim]
  row RS      : (acc | l) partials      [P_o chunk, dim + heads]   (the fused layer's combine)
and, to get the cost curve of the primitive on this layout, the column gather with SYNTHETIC need sets of share
f in --shares (random rows of the column block; the real share is one point of that curve).  Every number is the max
over the ranks (all groups run concurrently, like in training) of the mean over --iters calls.

    GT3D_SCRIPT=bench/bench_sparse_comm.py srun -N 4 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh \\
        --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --grid 4 4 --hidden 128
Prints SPARSE lines (one per collective and hidden size) and CURVE lines (synthetic shares).
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist.grid2d import Grid2D, raw_all_gather_rows, raw_reduce_scatter_rows  # noqa: E402
from gt3d.dist.sparse_attention import AttentionBlock  # noqa: E402
from gt3d.dist.sparse_comm import SparseRowPlan, tile_need_sets, group_node_layout  # noqa: E402
from gt3d.data.expander import block_expander_edges  # noqa: E402


def timeit(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return float(ms.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--hidden", type=int, nargs="+", default=[80, 128, 256])
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--shares", type=float, nargs="*", default=[0.03, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    ap.add_argument("--no_expander", action="store_true")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    plexus_fmt = os.path.exists(os.path.join(args.data, "metadata.pt"))
    if plexus_fmt:
        from gt3d.data.plexus_loader import load_plexus_partition as load_partition, read_plexus_meta, grid_n_pad
        meta = read_plexus_meta(args.data); meta["n_pad"] = grid_n_pad(meta, world, 4096)
    else:
        from gt3d.data.loader import load_partition, read_meta
        meta = read_meta(args.data)
    n = meta["num_nodes"]
    grid = Grid2D(args.grid[0], args.grid[1], n, n_pad=meta["n_pad"]).to(dev)
    part = load_partition(args.data, grid, dev, load_weights=False)
    blk = part["edge_index"].to(dev)
    if not args.no_expander:
        blk = torch.cat([blk, block_expander_edges(grid, n, args.exp_deg, seed=args.seed, device=dev)], dim=1)
    tile = AttentionBlock(grid, blk, None)
    del part, blk
    n_rows, (c_lo, c_hi) = grid.n_row_block, grid.col_block_range()
    n_cols = c_hi - c_lo
    g_row, g_col = dist.get_world_size(grid.row_group), dist.get_world_size(grid.col_group)
    rows, cols = tile_need_sets(tile.crow, tile.col, n_cols)
    plan_row = SparseRowPlan(grid.row_group, n_rows // g_row, rows, name="row")
    plan_col = SparseRowPlan(grid.col_group, n_cols // g_col, cols, name="col")
    lay_row, lay_col = group_node_layout(grid.row_group), group_node_layout(grid.col_group)
    name = os.path.basename(args.data.rstrip("/"))
    gs = f"{args.grid[0]}x{args.grid[1]}"
    if rank == 0:
        print(f"[bench] {name} grid {gs} gpus {world}: tile {tile.num_edges} edges, row block {n_rows} rows (need {plan_row.n_need}, "
              f"share {plan_row.share:.3f}, group {g_row} ranks, {lay_row[0]}/node{' multi-node' if lay_row[1] else ''}), "
              f"column block {n_cols} (need {plan_col.n_need}, share {plan_col.share:.3f}, group {g_col} ranks, "
              f"{lay_col[0]}/node{' multi-node' if lay_col[1] else ''})", flush=True)
    cdt = torch.bfloat16
    it = args.iters

    def report(tag, dim, dense_ms, sparse_ms, dense_rows, sparse_rows, plan, group_tag):
        db, sb = dense_rows * dim * 2, sparse_rows * dim * 2
        if rank == 0:
            print(f"SPARSE data={name} grid={gs} gpus={world} coll={tag} group={group_tag} dim={dim} share={plan.share:.3f} "
                  f"dense_ms={dense_ms:.3f} sparse_ms={sparse_ms:.3f} speedup={dense_ms / max(sparse_ms, 1e-9):.2f} "
                  f"dense_MB={db / 1e6:.1f} sparse_MB={sb / 1e6:.1f} dense_GBs={db / dense_ms / 1e6:.1f} sparse_GBs={sb / max(sparse_ms, 1e-9) / 1e6:.1f}",
                  flush=True)

    for dim in args.hidden:
        chunk = grid.chunk
        x = torch.randn(chunk, dim, device=dev).to(cdt)
        # column gather / reduce-scatter
        d_ms = timeit(lambda: raw_all_gather_rows(x, grid.col_group), it)
        s_ms = timeit(lambda: plan_col.gather(x), it)
        report("col_gather", dim, d_ms, s_ms, (g_col - 1) * chunk, plan_col.n_need - plan_col.recv_counts[plan_col.me], plan_col,
               f"col{g_col}:{lay_col[0]}pn")
        y_d = torch.empty(n_cols, dim, device=dev, dtype=cdt).normal_()
        y_s = torch.empty(max(plan_col.n_need, 1), dim, device=dev, dtype=cdt).normal_()
        d_ms = timeit(lambda: raw_reduce_scatter_rows(y_d, grid.col_group), it)
        s_ms = timeit(lambda: plan_col.reduce_scatter(y_s), it)
        report("col_rs", dim, d_ms, s_ms, (g_col - 1) * chunk, plan_col.n_need - plan_col.recv_counts[plan_col.me], plan_col,
               f"col{g_col}:{lay_col[0]}pn")
        del y_d, y_s
        # row gather / partials reduce-scatter
        d_ms = timeit(lambda: raw_all_gather_rows(x, grid.row_group), it)
        s_ms = timeit(lambda: plan_row.gather(x), it)
        report("row_gather", dim, d_ms, s_ms, (g_row - 1) * chunk, plan_row.n_need - plan_row.recv_counts[plan_row.me], plan_row,
               f"row{g_row}:{lay_row[0]}pn")
        dp = dim + args.heads
        y_d = torch.empty(n_rows, dp, device=dev, dtype=cdt).normal_()
        y_s = torch.empty(max(plan_row.n_need, 1), dp, device=dev, dtype=cdt).normal_()
        d_ms = timeit(lambda: raw_reduce_scatter_rows(y_d, grid.row_group), it)
        s_ms = timeit(lambda: plan_row.reduce_scatter(y_s), it)
        report("row_rs", dp, d_ms, s_ms, (g_row - 1) * chunk, plan_row.n_need - plan_row.recv_counts[plan_row.me], plan_row,
               f"row{g_row}:{lay_row[0]}pn")
        del y_d, y_s, x
    # cost curve of the column gather on this layout: synthetic need sets of share f (random rows, same f on every rank)
    dim = args.hidden[-1]
    x = torch.randn(grid.chunk, dim, device=dev).to(cdt)
    d_ms = timeit(lambda: raw_all_gather_rows(x, grid.col_group), it)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    for f in args.shares:
        k = max(1, int(round(f * n_cols)))
        need = torch.sort(torch.randperm(n_cols, generator=gen)[:k])[0].to(dev)
        p = SparseRowPlan(grid.col_group, grid.chunk, need, name=f"syn{f}")
        s_ms = timeit(lambda: p.gather(x), it)
        y_c = torch.empty(max(p.n_need, 1), dim, device=dev, dtype=cdt).normal_()
        r_ms = timeit(lambda: p.reduce_scatter(y_c), it)
        del y_c
        if rank == 0:
            print(f"CURVE data={name} grid={gs} gpus={world} group=col{g_col}:{lay_col[0]}pn dim={dim} share={f:.3f} "
                  f"dense_ms={d_ms:.3f} sparse_gather_ms={s_ms:.3f} sparse_rs_ms={r_ms:.3f} speedup={d_ms / max(s_ms, 1e-9):.2f}", flush=True)
        del p, need
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
