"""Distributed (Grid2D) attention vs the single-process reference.

    torchrun --nproc_per_node 4 tests/test_dist_attention.py --grid 2 2 --backend torch --dtype fp64
    torchrun --nproc_per_node 4 tests/test_dist_attention.py --grid 4 1 --backend triton --dtype fp32
    torchrun --nproc_per_node 4 tests/test_dist_attention.py --grid 4 1 --ring_kv --backend torch --dtype fp64

Every rank builds the same random global graph (seeded), the reference is computed on the padded
global graph, and the owned chunk of o and of dq/dk/dv/dgate is compared."""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.ref import masked_sparse_attention_ref  # noqa: E402
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.dist.sparse_attention import DistMaskedSparseAttention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[2, 2], help="P_d P_o [P_i]; with P_i: Grid3D + head axis")
    ap.add_argument("--backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--dtype", default="fp64", choices=["fp64", "fp32"])
    ap.add_argument("--norm", default="clampexp", choices=["clampexp", "softmax"])
    ap.add_argument("--score", default="gated", choices=["gated", "dot"])
    ap.add_argument("--num_nodes", type=int, default=1000)
    ap.add_argument("--num_edges", type=int, default=20000)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=40)
    ap.add_argument("--ring_kv", action="store_true", help="ring attention with K/V rotation (gt3d/dist/ring_kv.py) on a (P, 1) grid")
    ap.add_argument("--bidir", action="store_true", help="with --ring_kv: bidirectional ring (half A clockwise, half B counter-clockwise)")
    args = ap.parse_args()
    if args.ring_kv:
        assert len(args.grid) == 2 and args.grid[1] == 1 and args.norm == "clampexp", "--ring_kv needs --grid P 1 and clampexp"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    dtype = torch.float64 if args.dtype == "fp64" else torch.float32

    if len(args.grid) == 3:
        from gt3d.dist.grid3d import Grid3D
        from gt3d.dist.sparse_attention import DistHeadSparseAttention
        grid = Grid3D(args.grid[0], args.grid[1], args.grid[2], args.num_nodes)
    else:
        grid = Grid2D(args.grid[0], args.grid[1], args.num_nodes)
    n_pad = grid.n_pad
    gen = torch.Generator().manual_seed(0)
    ei = torch.randint(0, args.num_nodes, (2, args.num_edges), generator=gen)
    hub = torch.stack([torch.arange(args.num_nodes), torch.zeros(args.num_nodes, dtype=torch.long)])   # node 0 attends to all
    ei = torch.cat([ei, hub], dim=1)
    etype = torch.randint(0, 2, (ei.shape[1],), generator=gen)
    q = torch.randn(n_pad, args.heads, args.dim, generator=gen, dtype=dtype)
    k = torch.randn(n_pad, args.heads, args.dim, generator=gen, dtype=dtype)
    v = torch.randn(n_pad, args.heads, args.dim, generator=gen, dtype=dtype)
    gate = torch.randn(2, args.heads, args.dim, generator=gen, dtype=dtype)
    w = torch.randn(n_pad, args.heads, args.dim, generator=gen, dtype=dtype)     # loss weights

    # ---- reference on the padded global graph (identical on every rank) ----
    ref_in = [t.clone().to(dev).requires_grad_(True) for t in (q, k, v, gate)]
    o_ref = masked_sparse_attention_ref(ref_in[0], ref_in[1], ref_in[2], ei.to(dev), edge_type=etype.to(dev),
                                        gate=ref_in[3], score=args.score, norm=args.norm)
    (o_ref * w.to(dev)).sum().backward()

    # ---- distributed ----
    lo, hi = grid.owned_range()
    blk_ei, blk_et = grid.local_block(ei, etype)
    if args.ring_kv:
        from gt3d.dist.ring_kv import DistRingAttentionKV
        attn = DistRingAttentionKV.from_block(grid, blk_ei.to(dev), blk_et.to(dev), args.heads, args.dim,
                                              backend=args.backend, score=args.score, norm=args.norm, bidir=args.bidir)
    else:
        cls = DistHeadSparseAttention if len(args.grid) == 3 else DistMaskedSparseAttention
        attn = cls(grid, (blk_ei.to(dev), blk_et.to(dev)), args.heads, args.dim,
                   backend=args.backend, score=args.score, norm=args.norm)
    loc_in = [t[lo:hi].clone().to(dev).requires_grad_(True) for t in (q, k, v)]
    gate_loc = gate.clone().to(dev).requires_grad_(True)
    o_loc = attn(loc_in[0], loc_in[1], loc_in[2], gate_loc)
    (o_loc * w[lo:hi].to(dev)).sum().backward()

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()

    errs = {"o": rel(o_loc, o_ref[lo:hi])}
    for name, a, b in zip(["q", "k", "v"], loc_in, ref_in[:3]):
        errs[name] = rel(a.grad, b.grad[lo:hi])
    if args.score == "gated":
        g_total = gate_loc.grad.clone()
        dist.all_reduce(g_total)                     # the module returns the rank's partial
        errs["gate"] = rel(g_total, ref_in[3].grad)
    tol = 1e-10 if args.dtype == "fp64" else 2e-4
    ok = all(e < tol for e in errs.values())
    print(f"[dist rank {rank} r={grid.r} c={grid.c}{' h=%d' % grid.h if len(args.grid) == 3 else ''}] grid={tuple(args.grid)}{(' ring_kv2' if args.bidir else ' ring_kv') if args.ring_kv else ''} backend={args.backend} {args.dtype} "
          f"{args.score}/{args.norm}  " + "  ".join(f"{k}={v:.1e}" for k, v in errs.items()) + ("  OK" if ok else "  FAIL"),
          flush=True)
    flag = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(flag)
    dist.barrier()
    if rank == 0:
        print("[dist] ALL PASSED" if flag.item() == 0 else "[dist] FAILED", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() == 0 else 1)


if __name__ == "__main__":
    main()
