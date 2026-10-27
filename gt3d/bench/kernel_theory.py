"""Microbenchmarks behind the analytic kernel model (single GPU).

The per-row attention kernels move a known number of bytes per edge, per row and per column, so their time is the
larger of (a) the bytes over the HBM bandwidth and (b) the latency of the program's chain of dependent loads.  This
script measures what the model needs and what checks it:

  * HBM read bandwidth: a 1 GiB streaming reduction;
  * the per-program latency floor: time per row of every pass on REGULAR blocks (every row and every column has
    exactly d edges, d = 0 ... 128), so the time per row is flat (latency-bound) for short rows and grows with
    the bytes of the row for long rows;
  * the autotuned launch configuration and the register count of every kernel (occupancy).

    srun --jobid=<id> -n 1 bash scripts/run_on_node.sh python bench/kernel_theory.py
Writes bench/kernel_theory<tag>.json.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def hbm_read_gbs(dev):
    x = torch.randn(2 ** 28, device=dev)            # 1 GiB fp32
    t = timeit(lambda: x.sum(), iters=20)
    return x.numel() * 4 / (t * 1e-3) / 1e9


def regular_block(n, d, gen, dev):
    """n x n block in which every row and every column holds exactly d edges (d random permutations)."""
    if d == 0:
        crow = torch.zeros(n + 1, dtype=torch.long, device=dev)
        col = torch.zeros(0, dtype=torch.long, device=dev)
        et = torch.zeros(0, dtype=torch.long, device=dev)
    else:
        rows = torch.arange(n, device=dev).repeat(d)
        cols = torch.cat([torch.randperm(n, device=dev, generator=gen) for _ in range(d)])
        et0 = torch.randint(0, 2, (n * d,), device=dev, generator=gen)
        crow, col, et, _ = build_csr(torch.stack([cols, rows]), n, et0)
    return crow.to(torch.int32).contiguous(), col.to(torch.int32).contiguous(), et.to(torch.int32).contiguous()


def kernels(q, k, v, crow, col, et, gate, m0, dacc, dl, score="gated", tiled=False):
    args = tk._partial_args(q, k, v, m0, dacc, dl, None)
    q_, k_, v_, m_, dacc_, neg_dl, ones, scale = args
    g = gate if score == "gated" else None
    if tiled:
        from gt3d.attention import triton_tiled as tt
        kw = dict(score=score, norm="clampexp", clamp=(-5.0, 5.0), scale=scale)
        fwd = lambda: tt.attn_fwd_partial(q, k, v, crow, col, et, g, None, **kw)
        row = lambda: tt.bwd_row_partial(q_, k_, v_, crow, col, et, g, None, m_, dacc_, neg_dl, **kw)
        colp = lambda: tt.bwd_col_partial(q_, k_, v_, crow, col, et, g, None, m_, dacc_, neg_dl, **kw)
        return dict(fwd=fwd, row=row, col=colp)
    kw = dict(score=score, norm="clampexp", clamp=(-5.0, 5.0), eps=0.0, scale=scale, seg_edges=512)
    fwd = lambda: tk._attn_fwd_partial_rows(q, k, v, crow, col, et, g, None, score=score, norm="clampexp", seg_edges=512)
    row = lambda: tk._bwd_row_pass_rows(q_, k_, v_, crow, col, et, g, None, neg_dl, dacc_, ones, m_, partial=True, **kw)
    colp = lambda: tk._bwd_col_pass_rows(q_, k_, v_, crow, col, et, g, None, dacc_, ones, m_, neg_dl, **kw)
    return dict(fwd=fwd, row=row, col=colp)


def compiled_info(jit_fn):
    """Best autotuned config and register / spill counts of the compiled variants of an autotuned Triton kernel."""
    out = {}
    try:
        bc = jit_fn.best_config
        out["best"] = dict(kwargs=dict(bc.kwargs), num_warps=bc.num_warps, num_stages=bc.num_stages)
    except Exception as e:  # noqa: BLE001
        out["best_err"] = repr(e)
    try:
        fn = jit_fn.fn
        caches = fn.device_caches if hasattr(fn, "device_caches") else fn.cache
        variants = []
        for dev_cache in caches.values():
            entries = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
            for key, ker in entries.items():
                md = getattr(ker, "metadata", None)
                variants.append(dict(key=str(key)[:200], n_regs=getattr(ker, "n_regs", None), n_spills=getattr(ker, "n_spills", None),
                                     num_warps=getattr(md, "num_warps", None), shared=getattr(md, "shared", None)))
        out["variants"] = variants
    except Exception as e:  # noqa: BLE001
        out["variants_err"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="4x32,2x32,1x32,8x32,2x40,1x40,2x20")
    ap.add_argument("--n", type=int, default=612_352)
    ap.add_argument("--degrees", default="0,1,2,3,4,6,8,12,16,24,32,48,64,96,128")
    ap.add_argument("--small_n", default="58240,7280", help="L2-resident block sizes, a few degrees each")
    ap.add_argument("--variants", default="dot:4x32,tiled:4x32,tiled:2x32,tiled:1x32,tiled:8x32",
                    help="score=dot (no gate-gradient atomics) and tiled-kernel sweeps on regular blocks")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    say = lambda *a: print(*a, flush=True)
    out = dict(gpu=torch.cuda.get_device_name(), sm=torch.cuda.get_device_properties(dev).multi_processor_count)
    out["hbm_read_gbs"] = hbm_read_gbs(dev)
    say(f"[theory] {out['gpu']}  SMs {out['sm']}  HBM read {out['hbm_read_gbs']:.0f} GB/s")
    degrees = [int(x) for x in args.degrees.split(",")]
    out["regular"] = {}
    for cfg in args.configs.split(","):
        h, dh = map(int, cfg.split("x"))
        gate = torch.randn(2, h, dh, device=dev)
        res = {}
        for n, ds in [(args.n, degrees)] + [(int(s), [1, 4, 16, 64]) for s in args.small_n.split(",") if s]:
            q = torch.randn(n, h, dh, device=dev); k = torch.randn(n, h, dh, device=dev); v = torch.randn(n, h, dh, device=dev)
            dacc = torch.randn(n, h, dh, device=dev); dl = torch.randn(n, h, device=dev); m0 = torch.zeros(n, h, device=dev)
            # autotune on a typical block (16 edges per row) before the sweep, like a training run does on its block
            blk = regular_block(n, 16, gen, dev)
            for f in kernels(q, k, v, *blk, gate, m0, dacc, dl).values():
                f()
            torch.cuda.synchronize()
            del blk
            for d in ds:
                blk = regular_block(n, d, gen, dev)
                ks = kernels(q, k, v, *blk, gate, m0, dacc, dl)
                t = {name: timeit(f) for name, f in ks.items()}
                res[f"{n}_{d}"] = dict(n=n, d=d, **t)
                say(f"[theory] {cfg} n {n:7d} d {d:4d}: fwd {t['fwd']:8.3f}  row {t['row']:8.3f}  col {t['col']:8.3f} ms   "
                    f"per row: {t['fwd']*1e6/n:6.2f} {t['row']*1e6/n:6.2f} {t['col']*1e6/n:6.2f} ns")
                del blk, ks
                torch.cuda.empty_cache()
            del q, k, v, dacc, dl, m0
            torch.cuda.empty_cache()
        res["compiled"] = {name: compiled_info(fn) for name, fn in
                           (("fwd", tk._attn_fwd_kernel), ("row", tk._attn_bwd_kernel), ("col", tk._attn_bwd_kv_kernel))}
        say(f"[theory] {cfg} compiled: " + json.dumps(res["compiled"])[:1500])
        out["regular"][cfg] = res
    # variants: per-row kernels without the gate (no atomics in the row pass), and the tiled kernels
    out["variants"] = {}
    for spec in [x for x in args.variants.split(",") if x]:
        kind, cfg = spec.split(":")
        h, dh = map(int, cfg.split("x"))
        n = args.n
        gate = torch.randn(2, h, dh, device=dev)
        q = torch.randn(n, h, dh, device=dev); k = torch.randn(n, h, dh, device=dev); v = torch.randn(n, h, dh, device=dev)
        dacc = torch.randn(n, h, dh, device=dev); dl = torch.randn(n, h, device=dev); m0 = torch.zeros(n, h, device=dev)
        ds = [1, 2, 4, 8, 16, 32, 64] if kind == "dot" else [1, 2, 3, 4, 6, 8, 16]
        score, tiled = ("dot", False) if kind == "dot" else ("gated", True)
        blk = regular_block(n, ds[len(ds) // 2], gen, dev)
        for f in kernels(q, k, v, *blk, gate, m0, dacc, dl, score=score, tiled=tiled).values():
            f()
        del blk
        res = {}
        for d in ds:
            blk = regular_block(n, d, gen, dev)
            t = {name: timeit(f) for name, f in kernels(q, k, v, *blk, gate, m0, dacc, dl, score=score, tiled=tiled).items()}
            res[str(d)] = dict(n=n, d=d, **t)
            say(f"[theory] {spec} n {n:7d} d {d:4d}: fwd {t['fwd']:8.3f}  row {t['row']:8.3f}  col {t['col']:8.3f} ms   "
                f"per row: {t['fwd']*1e6/n:6.2f} {t['row']*1e6/n:6.2f} {t['col']*1e6/n:6.2f} ns")
            del blk
            torch.cuda.empty_cache()
        out["variants"][spec] = res
        del q, k, v, dacc, dl, m0
        torch.cuda.empty_cache()
    # per-call floor: a tiny block (CPU dispatch + launch of the wrapper's kernels)
    out["call_floor"] = {}
    for cfg in ("4x32", "1x32"):
        h, dh = map(int, cfg.split("x"))
        n = 1024
        gate = torch.randn(2, h, dh, device=dev)
        q = torch.randn(n, h, dh, device=dev); k = torch.randn(n, h, dh, device=dev); v = torch.randn(n, h, dh, device=dev)
        dacc = torch.randn(n, h, dh, device=dev); dl = torch.randn(n, h, device=dev); m0 = torch.zeros(n, h, device=dev)
        blk = regular_block(n, 4, gen, dev)
        t = {name: timeit(f, iters=50) for name, f in kernels(q, k, v, *blk, gate, m0, dacc, dl).items()}
        out["call_floor"][cfg] = t
        say(f"[theory] call floor {cfg} (n 1024, d 4): fwd {t['fwd']*1e3:.0f}  row {t['row']*1e3:.0f}  col {t['col']*1e3:.0f} us")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"kernel_theory{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    say(f"[theory] written {path}")


if __name__ == "__main__":
    main()
