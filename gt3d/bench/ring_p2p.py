"""Point-to-point cost of one ring hop, measured once in isolation (constants of the ring terms of the white-box model,
gt3d/dist/grid_select.py).  Every rank posts {isend to rank+1, irecv from rank-1} as one batched NCCL group, exactly as
gt3d/dist/ring.py does; `bufs` groups are in flight at once (1 = a forward hop, 2 = a backward step: the x chunk and
the dx accumulator).  Time per hop = max over ranks of the mean over `iters` hops.  Prints one P2P line per point.

    GT3D_SCRIPT=bench/ring_p2p.py srun --jobid=<id> -N <nodes> --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh
"""
import os

import torch
import torch.distributed as dist

from gt3d.dist.ring import RingComm


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank, world = dist.get_rank(), dist.get_world_size()
    nodes = max(1, world // min(world, int(os.environ.get("SLURM_NTASKS_PER_NODE", "4"))))
    ring = RingComm(dist.group.WORLD)
    iters, warm = 10, 3
    for mb in (1, 4, 16, 32, 64, 128):
        n = mb * 2 ** 20 // 2                                    # bf16 elements
        for bufs in (1, 2):
            send = [torch.ones(n, dtype=torch.bfloat16, device="cuda") for _ in range(bufs)]
            recv = [torch.empty(n, dtype=torch.bfloat16, device="cuda") for _ in range(bufs)]
            for i in range(warm + iters):
                if i == warm:
                    torch.cuda.synchronize()
                    dist.barrier()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                works = [w for b in range(bufs) for w in ring.send_recv(send[b], recv[b])]
                RingComm.wait(works)
            t1.record()
            torch.cuda.synchronize()
            ms = torch.tensor([t0.elapsed_time(t1) / iters], device="cuda")
            dist.all_reduce(ms, op=dist.ReduceOp.MAX)
            if rank == 0:
                gbs = bufs * mb * 2 ** 20 / (ms.item() * 1e6)
                print(f"P2P world={world} nodes={nodes} mb={mb} bufs={bufs} ms_per_hop={ms.item():.3f} "
                      f"gbs_per_rank={gbs:.1f}", flush=True)
            del send, recv
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
