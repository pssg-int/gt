"""Cross-node transport sanity: one contiguous 8-row block from the peer, host-initiated (getmem_on_stream) and
device-initiated (one thread, nvshmem_getmem), then the warp-level gather kernel."""
import ctypes, os, sys, socket
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist import pullcomm
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count()); dev = torch.device("cuda")
say = lambda *a: print(f"[rank {rank} {socket.gethostname()}]", *a, flush=True)  # noqa: E731
pullcomm.init(); L = pullcomm.lib()
L.pull_get_block_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
L.pull_get_block_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
R, F = 1024, 128
x = pullcomm.sym_empty((R, F), torch.bfloat16); x.fill_(float(rank + 1)); torch.cuda.synchronize()
pullcomm.barrier(); torch.cuda.synchronize(); say("barrier ok")
peer = (rank + 1) % world
dst = torch.zeros(8, F, dtype=torch.bfloat16, device=dev)
s = torch.cuda.current_stream().cuda_stream
L.pull_get_block_host(dst.data_ptr(), x.data_ptr(), 8 * F * 2, peer, s); torch.cuda.synchronize()
say("host-initiated get from PE", peer, "->", dst[0, 0].item(), f"(expect {peer + 1})")
dst.zero_(); L.pull_get_block_device(dst.data_ptr(), x.data_ptr(), 8 * F * 2, peer, s); torch.cuda.synchronize()
say("device-initiated (1 thread) get from PE", peer, "->", dst[0, 0].item())
src_pe = torch.full((8,), peer, dtype=torch.int32, device=dev); src_row = torch.arange(8, dtype=torch.int32, device=dev)
dst.zero_(); pullcomm.gather(dst, x, src_pe, src_row); torch.cuda.synchronize()
say("warp gather kernel from PE", peer, "->", dst[:, 0].tolist()[:3])
pullcomm.barrier(); torch.cuda.synchronize(); say("done")
dist.barrier(); dist.destroy_process_group()
