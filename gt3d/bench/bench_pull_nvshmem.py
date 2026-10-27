"""NVSHMEM receiver-driven sparse gather / fused reduce (gt3d/dist/pullcomm) vs NCCL dense collectives and the a2a-v
plan, on the ranks of the job as one group (run with one task per node for a cross-node column-group layout, four per
node for an intra-node group).  Synthetic need sets of density rho; correctness against the a2a-v plan.

    GT3D_SCRIPT=bench/bench_pull_nvshmem.py srun -N 4 --ntasks-per-node 1 --gpus-per-node 1 bash scripts/run_nvshmem.sh --rows 153088 --dims 128 256 --densities 0.2 0.45 0.8 1.0
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist import pullcomm  # noqa: E402
from gt3d.dist.sparse_comm import SparseRowPlan  # noqa: E402
from gt3d.dist.grid2d import raw_all_gather_rows, raw_reduce_scatter_rows  # noqa: E402


def timeit(fn, iters, warm=3, barrier=False):
    for _ in range(warm):
        if barrier:
            pullcomm.barrier()
        fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        if barrier:
            pullcomm.barrier()
        fn()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return float(ms.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=153088)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--densities", type=float, nargs="*", default=[0.2, 0.45, 0.8, 1.0])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    group = dist.group.WORLD
    import socket
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    pullcomm.init()
    R = args.rows
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    say(f"[nvshmem] world {world} PEs on {len(set(hosts))} node(s), rows/rank {R}, torch {torch.__version__}")
    for F in args.dims:
        x = pullcomm.sym_empty((R, F), torch.bfloat16); x.normal_()
        for rho in args.densities:
            gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
            k = max(1, int(round(rho * world * R)))
            need = torch.sort(torch.randperm(world * R, generator=gen)[:k])[0]
            needs = [None] * world
            dist.all_gather_object(needs, need.numpy())
            needs = [torch.as_tensor(n, dtype=torch.long) for n in needs]
            plan = SparseRowPlan(group, R, need.to(dev), name=f"rho{rho}")
            n_need = plan.n_need
            n_max = torch.tensor([n_need], device=dev); dist.all_reduce(n_max, op=dist.ReduceOp.MAX); n_max = int(n_max.item())
            owner = need // R
            src_pe = owner.to(torch.int32).to(dev)
            src_row = (need - owner * R).to(torch.int32).to(dev)
            # reduce tables (CSR over my owned rows): entries (pe, position in pe's compact buffer)
            lo, hi = rank * R, (rank + 1) * R
            ents = []
            for s in range(world):
                ns = needs[s]
                pos = torch.nonzero((ns >= lo) & (ns < hi)).squeeze(1)
                ents.append(torch.stack([ns[pos] - lo, torch.full_like(pos, s), pos], 1))
            ents = torch.cat(ents, 0)
            ents = ents[torch.argsort(ents[:, 0] * world + ents[:, 1])]
            row_ptr = torch.zeros(R + 1, dtype=torch.int32)
            row_ptr[1:] = torch.cumsum(torch.bincount(ents[:, 0], minlength=R), 0).to(torch.int32)
            row_ptr, ent_pe, ent_idx = row_ptr.to(dev), ents[:, 1].to(torch.int32).to(dev), ents[:, 2].to(torch.int32).to(dev)
            part = pullcomm.sym_empty((n_max, F), torch.bfloat16); part.normal_()
            out_g = torch.empty(n_need, F, dtype=torch.bfloat16, device=dev)
            out_r = torch.empty(R, F, dtype=torch.bfloat16, device=dev)
            # correctness
            pullcomm.barrier()
            ref_g = plan.gather(x)
            pullcomm.gather(out_g, x, src_pe, src_row)
            err_g = (out_g.float() - ref_g.float()).abs().max().item()
            out_g.zero_(); pullcomm.gather(out_g, x, src_pe, src_row, nbi=True)
            err_g2 = (out_g.float() - ref_g.float()).abs().max().item()
            ref_r = plan.reduce_scatter(part[:n_need].float())
            pullcomm.reduce(out_r, part, row_ptr, ent_pe, ent_idx)
            err_r = ((out_r.float() - ref_r).abs().max() / ref_r.abs().max().clamp_min(1e-6)).item()
            e = torch.tensor([err_g, err_g2, err_r], device=dev); dist.all_reduce(e, op=dist.ReduceOp.MAX)
            y_full = torch.empty(world * R, F, dtype=torch.bfloat16, device=dev).normal_()
            t_ag = timeit(lambda: raw_all_gather_rows(x, group), args.iters)
            t_rs = timeit(lambda: raw_reduce_scatter_rows(y_full, group), args.iters)
            t_pg = timeit(lambda: plan.gather(x), args.iters)
            t_pr = timeit(lambda: plan.reduce_scatter(part[:n_need]), args.iters)
            t_ng = timeit(lambda: pullcomm.gather(out_g, x, src_pe, src_row), args.iters, barrier=True)
            t_nn = timeit(lambda: pullcomm.gather(out_g, x, src_pe, src_row, nbi=True), args.iters, barrier=True)
            t_nr = timeit(lambda: pullcomm.reduce(out_r, part, row_ptr, ent_pe, ent_idx), args.iters, barrier=True)
            mb_dense = (world - 1) * R * F * 2 / 1e6
            mb_sparse = (n_need - int((owner == rank).sum())) * F * 2 / 1e6
            say(f"NVSHMEM pes={world} nodes={len(set(hosts))} rows={R} dim={F} density={rho:.2f} dense_MB={mb_dense:.0f} sparse_MB={mb_sparse:.0f} "
                f"| gather: nccl_AG {t_ag:.2f} ms  a2av {t_pg:.2f}  pull {t_ng:.2f}  pull_nbi {t_nn:.2f}  "
                f"| reduce: nccl_RS {t_rs:.2f}  a2av+spmm {t_pr:.2f}  pull_fused {t_nr:.2f}  "
                f"| check: gather {e[0].item():.1e}/{e[1].item():.1e} reduce {e[2].item():.1e}")
            pullcomm.free(part)
            del part, out_g, out_r, y_full, plan
            torch.cuda.synchronize()
        pullcomm.free(x); del x
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
