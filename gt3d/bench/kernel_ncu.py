"""Workload for Nsight Compute: the three passes of the attention kernel on the regular blocks of bench/kernel_theory.py
(every row and column holds d edges), one call per (configuration, pass, degree) between cudaProfilerStart / Stop, so
that `ncu --profile-from-start off --kernel-name regex:_attn_` captures exactly these launches (autotuning happens
before, outside the profiled range).  The launch order is printed; bench/kernel_ncu_report.py joins it with the report.

    scripts/run_kernel_ncu.sh   (whole node, dcgmi profile --pause / --resume around ncu)
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk  # noqa: E402
from bench.kernel_theory import regular_block, kernels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="gated:4x32:1,4,16,64,128;dot:4x32:1,16,64;gated:2x32:1,64;gated:1x32:1,64")
    ap.add_argument("--n", type=int, default=612_352)
    ap.add_argument("--order", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_ncu_order.json"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    order = []
    plan = []
    for case in args.cases.split(";"):
        score, cfg, ds = case.split(":")
        plan.append((score, cfg, [int(x) for x in ds.split(",")]))
    for score, cfg, ds in plan:
        h, dh = map(int, cfg.split("x"))
        n = args.n
        gate = torch.randn(2, h, dh, device=dev)
        q = torch.randn(n, h, dh, device=dev); k = torch.randn(n, h, dh, device=dev); v = torch.randn(n, h, dh, device=dev)
        dacc = torch.randn(n, h, dh, device=dev); dl = torch.randn(n, h, device=dev); m0 = torch.zeros(n, h, device=dev)
        for mod in (tk._attn_fwd_kernel, tk._attn_bwd_kernel, tk._attn_bwd_kv_kernel):
            mod.cache.clear()
        blk = regular_block(n, 16, gen, dev)                 # tune on a typical block, as a training run does
        for f in kernels(q, k, v, *blk, gate, m0, dacc, dl, score=score).values():
            f()
        del blk
        for d in ds:
            blk = regular_block(n, d, gen, dev)
            ks = kernels(q, k, v, *blk, gate, m0, dacc, dl, score=score)
            for f in ks.values():                            # warm: plans, CSC, caches
                f()
            torch.cuda.synchronize()
            for name, f in ks.items():
                torch.cuda.profiler.start()
                f()
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                order.append(dict(score=score, cfg=cfg, d=d, n=n, pass_=name))
                print(f"[ncu-order] {len(order) - 1:3d} {score} {cfg} d={d} {name}", flush=True)
            del blk, ks
            torch.cuda.empty_cache()
        del q, k, v, dacc, dl, m0
        torch.cuda.empty_cache()
    json.dump(order, open(args.order, "w"), indent=1)
    print(f"[ncu-order] written {args.order}", flush=True)


if __name__ == "__main__":
    main()
