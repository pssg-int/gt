"""Plexus partition reader vs gt3d's own partition of the SAME processed graph (Reddit): owned features, labels,
masks and the local attention/GCN block (edge set + weights) must agree on every rank.  CPU/gloo.

    torchrun --nproc_per_node 4 tests/test_plexus_loader.py --grid 2 2
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.data.loader import load_partition, read_meta  # noqa: E402
from gt3d.data.plexus_loader import load_plexus_partition, read_plexus_meta  # noqa: E402

GT3D = "/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16"
PLEXUS = "/pscratch/sd/c/cunyang/gnn/plexus/dataset/reddit/processed_part4"


def edge_key(ei, w):
    order = torch.argsort(ei[1].long() * (1 << 32) + ei[0].long())
    return ei[:, order], (w[order] if w is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    args = ap.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    meta = read_meta(GT3D)
    pm = read_plexus_meta(PLEXUS)
    grid = Grid2D(args.grid[0], args.grid[1], meta["num_nodes"], n_pad=meta["n_pad"])
    ref = load_partition(GT3D, grid, torch.device("cpu"))
    results = {}
    for src_first in (False, True):
        got = load_plexus_partition(PLEXUS, grid, torch.device("cpu"), src_first=src_first)
        ok_x = torch.equal(got["x"], ref["x"]) and torch.equal(got["y"], ref["y"])
        ok_m = all(torch.equal(got["masks"][k], ref["masks"][k]) for k in ref["masks"])
        e_ref, w_ref = edge_key(ref["edge_index"], ref["weight"])
        e_got, w_got = edge_key(got["edge_index"], got["weight"])
        ok_e = e_ref.shape == e_got.shape and torch.equal(e_ref, e_got)
        ok_w = ok_e and torch.allclose(w_ref, w_got)
        results[src_first] = (ok_x, ok_m, ok_e, ok_w, int(got["num_local_edges"]), int(ref["num_local_edges"]))
    flags = torch.tensor([[int(v) for v in results[sf][:4]] for sf in (False, True)])
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"plexus meta {pm}")
        for sf in (False, True):
            print(f"src_first={sf}: x/y {bool(flags[int(sf), 0])} masks {bool(flags[int(sf), 1])} edges {bool(flags[int(sf), 2])} "
                  f"weights {bool(flags[int(sf), 3])}  (rank0 local edges plexus {results[sf][4]} vs gt3d {results[sf][5]})")
        good = bool(flags[0].all()) or bool(flags[1].all())
        which = "edge_index[0]=dst (Plexus A[dst,src])" if bool(flags[0].all()) else ("edge_index[0]=src (PyG)" if bool(flags[1].all()) else "none")
        print(f"[plexus-loader] {'ALL PASSED' if good else 'FAILED'}; matching orientation: {which}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
