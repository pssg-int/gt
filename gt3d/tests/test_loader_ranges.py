"""Partition loader with grid chunks SMALLER than partition chunks (world > p_part), CPU/gloo:
every rank loads its slice; the real feature rows and the local edges must add up to the partition's totals,
and every rank's x must equal the corresponding rows of the partition's x files.

    torchrun --nproc_per_node 32 tests/test_loader_ranges.py --data .../products_p16b --grid 8 4
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.data.loader import load_partition, read_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--grid", type=int, nargs=2, required=True)
    args = ap.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    meta = read_meta(args.data)
    n, pc = meta["num_nodes"], meta["chunk"]
    grid = Grid2D(args.grid[0], args.grid[1], n, n_pad=meta["n_pad"])
    part = load_partition(args.data, grid, torch.device("cpu"))
    lo, hi = grid.owned_range()
    n_real = max(0, min(hi, n) - lo)
    # x check against the partition files
    ok = True
    if n_real > 0:
        i0, i1 = lo // pc, (min(hi, n) - 1) // pc
        ref = torch.cat([torch.load(os.path.join(args.data, "x", f"{i}.pt"), weights_only=False) for i in range(i0, i1 + 1)])
        ref = ref[lo - i0 * pc: lo - i0 * pc + n_real]
        ok &= torch.equal(part["x"][:n_real], ref)
        ok &= bool((part["x"][n_real:] == 0).all())
    # edge ranges: src (col-local) < P_d*chunk, dst (row-local) < P_o*chunk
    ei = part["edge_index"]
    ok &= bool(ei[0].max() < grid.p_rows * grid.chunk) and bool(ei[1].max() < grid.p_cols * grid.chunk)
    tot = torch.tensor([n_real, part["num_local_edges"], int(part["masks"]["train"].sum()), 0 if ok else 1], dtype=torch.long)
    dist.all_reduce(tot)
    if rank == 0:
        print(f"nodes {tot[0].item()} (meta {n})  edges {tot[1].item()} (meta {meta['num_edges']})  train {tot[2].item()}  "
              f"x/range failures {tot[3].item()}", flush=True)
        good = tot[0].item() == n and tot[1].item() == meta["num_edges"] and tot[3].item() == 0
        print("[loader-ranges] ALL PASSED" if good else "[loader-ranges] FAILED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
