"""torch.profiler breakdown of SparseRowPlan.gather vs reduce_scatter on one node (4 GPUs): which kernels make the
sparse reduce-scatter slower than the sparse gather of the same bytes (RESULTS 2026-10-23, 64-GPU section)?
Synthetic plans with a given share on the (4, 1) grid's column group (4 ranks, NVLink) and a papers-like row group
case (rows_per_rank 1.7M, dim 132, 3 holders).  Rank 0 prints the device kernel time per kernel name.

    GT3D_SCRIPT=bench/profile_sparse_comm.py srun -N1 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh
"""
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist

from gt3d.dist.sparse_comm import SparseRowPlan


def profile(fn, name, rank, iters=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = defaultdict(float)
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            agg[ev.key[:60]] += ev.self_device_time_total / iters / 1e3
    if rank == 0:
        tot = sum(agg.values())
        print(f"PROFILE {name}: wall {wall:.2f} ms, device kernels {tot:.2f} ms: " +
              "; ".join(f"{k} {v:.2f}" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:8]), flush=True)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    group = dist.new_group(list(range(world)))
    cdt = torch.bfloat16
    gen = torch.Generator(device="cpu").manual_seed(rank)
    for rows_per_rank, dim, share in [(612352, 256, 1.0), (612352, 256, 0.6), (1735360, 132, 0.97), (1735360, 136, 0.97)]:
        n_blk = world * rows_per_rank
        k = int(round(share * n_blk))
        need = torch.sort(torch.randperm(n_blk, generator=gen)[:k])[0].to(dev)
        plan = SparseRowPlan(group, rows_per_rank, need, name=f"syn{share}")
        x = torch.empty(rows_per_rank, dim, device=dev, dtype=cdt).normal_()
        y = torch.empty(max(plan.n_need, 1), dim, device=dev, dtype=cdt).normal_()
        tag = f"rows/rank {rows_per_rank} dim {dim} share {share} (n_send {plan.n_send})"
        profile(lambda: plan.gather(x), f"gather  {tag}", rank)
        profile(lambda: plan.reduce_scatter(y), f"reduce-scatter {tag}", rank)
        recv = torch.empty(max(plan.n_send, 1), dim, device=dev, dtype=cdt).normal_()
        profile(lambda: plan._accumulate(recv[:plan.n_send]), f"accumulate only {tag}", rank)
        del plan, x, y, recv, need
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
