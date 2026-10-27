"""Does the autotuned launch configuration (chosen on the FIRST block a process sees) change the time of later blocks?
Autotunes on a regular block of --first edges per row, then times regular blocks of 4 ... 64 edges per row."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk  # noqa: E402
from bench.kernel_theory import regular_block, kernels, timeit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, default=57)
ap.add_argument("--cfg", default="4x32")
a = ap.parse_args()
dev = torch.device("cuda")
gen = torch.Generator(device=dev).manual_seed(0)
h, dh = map(int, a.cfg.split("x"))
n = 612352
gate = torch.randn(2, h, dh, device=dev)
q = torch.randn(n, h, dh, device=dev); k = torch.randn(n, h, dh, device=dev); v = torch.randn(n, h, dh, device=dev)
dacc = torch.randn(n, h, dh, device=dev); dl = torch.randn(n, h, device=dev); m0 = torch.zeros(n, h, device=dev)
blk = regular_block(n, a.first, gen, dev)
for f in kernels(q, k, v, *blk, gate, m0, dacc, dl).values():
    f()
del blk
cfgs = {name: (fn.best_config.kwargs, fn.best_config.num_warps) for name, fn in
        (("fwd", tk._attn_fwd_kernel), ("row", tk._attn_bwd_kernel), ("col", tk._attn_bwd_kv_kernel))}
print(f"[order] {a.cfg} autotuned on d={a.first}: {cfgs}", flush=True)
for d in (1, 4, 8, 16, 32, 64):
    blk = regular_block(n, d, gen, dev)
    t = {name: timeit(f) for name, f in kernels(q, k, v, *blk, gate, m0, dacc, dl).items()}
    print(f"[order] first={a.first} d={d:3d}: per row fwd {t['fwd']*1e6/n:6.2f} row {t['row']*1e6/n:6.2f} col {t['col']*1e6/n:6.2f} ns", flush=True)
    del blk
