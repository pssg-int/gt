"""Minimal cross-node NVSHMEM pull check: init, symmetric buffers, barrier, one small gather from every peer, one small
fused reduce; prints after each step (rank 0) so a crash is localised.  Run with one task per node on >= 2 nodes."""
import os, sys, socket
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist import pullcomm
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count()); dev = torch.device("cuda")
say = lambda *a: print(f"[rank {rank} {socket.gethostname()}]", *a, flush=True)  # noqa: E731
pullcomm.init(); say("init ok")
R, F = 1024, 128
x = pullcomm.sym_empty((R, F), torch.bfloat16); say("sym_empty ok", hex(x._pull_ptr))
x.fill_(float(rank + 1)); torch.cuda.synchronize(); say("fill ok")
pullcomm.barrier(); torch.cuda.synchronize(); say("barrier ok")
# gather row 0..7 of every peer
src_pe = torch.arange(world, device=dev, dtype=torch.int32).repeat_interleave(8)
src_row = torch.arange(8, device=dev, dtype=torch.int32).repeat(world)
out = torch.zeros(world * 8, F, dtype=torch.bfloat16, device=dev)
pullcomm.gather(out, x, src_pe, src_row); torch.cuda.synchronize()
say("gather ok:", out[:, 0].view(world, 8)[:, 0].tolist(), "(expect 1..world)")
pullcomm.gather(out, x, src_pe, src_row, nbi=True); torch.cuda.synchronize(); say("gather nbi ok:", out[:, 0].view(world, 8)[:, 0].tolist())
# fused reduce: my row i sums row i of every peer -> world*(world+1)/2
row_ptr = torch.arange(0, (8 + 1) * world, world, dtype=torch.int32, device=dev)
ent_pe = torch.arange(world, dtype=torch.int32, device=dev).repeat(8)
ent_idx = torch.arange(8, dtype=torch.int32, device=dev).repeat_interleave(world)
outr = torch.zeros(8, F, dtype=torch.bfloat16, device=dev)
pullcomm.reduce(outr, x, row_ptr, ent_pe, ent_idx); torch.cuda.synchronize()
say("reduce ok:", outr[:, 0].tolist()[:3], f"(expect {world*(world+1)//2})")
pullcomm.barrier(); torch.cuda.synchronize(); say("done")
dist.barrier(); dist.destroy_process_group()
