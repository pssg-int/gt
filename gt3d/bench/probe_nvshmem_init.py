"""Where does NVSHMEM init crash inside a torch process?  Stage 1: ctypes only (no torch).  Stage 2: after importing
torch and creating its CUDA context.  Stage 3: after torch.distributed init.  Each stage is selected by --stage so a
crash in one stage does not hide the others; nranks = 1."""
import argparse, ctypes, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1); ap.add_argument("--local", action="store_true"); args = ap.parse_args()
say = lambda *a: print(*a, flush=True)  # noqa: E731
say("env NVSHMEM_REMOTE_TRANSPORT =", os.environ.get("NVSHMEM_REMOTE_TRANSPORT"), " VMM off =", os.environ.get("NVSHMEM_DISABLE_CUDA_VMM"))
if args.stage >= 2:
    import torch
    torch.cuda.set_device(0); torch.cuda.init(); _ = torch.zeros(1, device="cuda")
    say(f"stage {args.stage}: torch {torch.__version__} context created")
if args.stage >= 3:
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=0, world_size=1)
    t = torch.zeros(1, device="cuda"); dist.all_reduce(t)
    say("stage 3: torch.distributed initialised (nccl)")
path = "/pscratch/sd/c/cunyang/gnn/gt/gt3d/gt3d/dist/pullcomm/libpullcomm.so"
L = ctypes.CDLL(path, mode=(ctypes.RTLD_LOCAL if args.local else ctypes.RTLD_GLOBAL))
L.pull_uid_size.restype = ctypes.c_int
L.pull_get_uid.argtypes = [ctypes.c_void_p]; L.pull_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
L.pull_malloc.argtypes = [ctypes.c_size_t]; L.pull_malloc.restype = ctypes.c_void_p
n = L.pull_uid_size(); buf = ctypes.create_string_buffer(n)
say(f"uid size {n}; get_uid rc={L.pull_get_uid(buf)}")
say("calling pull_init ...")
rc = L.pull_init(buf, 0, 1)
say(f"pull_init rc={rc} pe={L.pull_my_pe()}/{L.pull_n_pes()}")
p = L.pull_malloc(1 << 20); say(f"malloc {p:#x}")
L.pull_finalize(); say("finalize ok")
