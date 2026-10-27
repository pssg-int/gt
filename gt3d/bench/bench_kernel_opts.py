"""Kernel-side options on synthetic blocks (one GPU): heads per program (8 vs 4), bf16 Q / K / V / dacc operands,
and the per-pass dispatch of the backward (column pass tiled when the block has few edges per column).
    python bench/bench_kernel_opts.py
Blocks: products at 16 GPUs 4x4 (612K x 612K, 8.8M edges), products at 64 GPUs 16x4 (153K x 612K, 2.0M edges),
Reddit at 64 GPUs (14.6K x 58K, 1.8M edges); heads x d_head: 8x32 (h256), 4x32 (h128), 2x40 (h80)."""
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


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    blocks = [("products16", 612_352, 612_352, 8_837_500), ("products64", 153_088, 612_352, 1_976_184), ("reddit64", 14_592, 58_368, 1_779_295)]
    for name, R, C, E in blocks:
        src = torch.randint(0, C, (E,), device=dev, generator=gen)
        dst = torch.randint(0, R, (E,), device=dev, generator=gen)
        et = torch.randint(0, 2, (E,), device=dev, generator=gen)
        crow, col, etype, _ = build_csr(torch.stack([src, dst]), R, et)
        crow32, col32, et32 = crow.to(torch.int32).contiguous(), col.to(torch.int32), etype.to(torch.int32)
        for h, dh in ((8, 32), (4, 32), (2, 40)):
            q = torch.randn(R, h, dh, device=dev); k = torch.randn(C, h, dh, device=dev); v = torch.randn(C, h, dh, device=dev)
            gate = torch.randn(2, h, dh, device=dev); dacc = torch.randn(R, h, dh, device=dev); dl = torch.randn(R, h, device=dev)
            m0 = torch.zeros(R, h, device=dev)
            ref = None
            for hpp in (8, 4):
                if hpp > h:
                    continue
                for lowp in (False, True):
                    tk.HEADS_PER_PROGRAM = hpp
                    qq, kk, vv, da = (t.to(torch.bfloat16) if lowp else t for t in (q, k, v, dacc))
                    for col_tiled in ((False, True) if name == "products64" else (False,)):
                        thr = tk.TILED_THRESHOLD
                        if not col_tiled:
                            tk.TILED_THRESHOLD = 0.0                                   # per-row kernels for both passes
                        f = lambda: tk.attn_fwd_partial(qq, kk, vv, crow32, col32, et32, gate, None, score="gated", norm="clampexp")
                        b = lambda: tk.attn_bwd_partial(qq, kk, vv, crow32, col32, et32, gate, None, m0, da, dl, score="gated", norm="clampexp")
                        acc, l, _ = f(); dq, dk, dv, dg, _ = b()
                        if ref is None:
                            ref = (acc.clone(), l.clone(), dq.clone(), dk.clone(), dv.clone(), dg.clone())
                        err = max(float((x - y).abs().max() / (y.abs().max() + 1e-9)) for x, y in zip((acc, l, dq, dk, dv, dg), ref))
                        t_f, t_b = timeit(f), timeit(b)
                        tk.TILED_THRESHOLD = thr
                        print(f"{name:11s} h{h} d{dh}  heads/program {hpp}  {'bf16' if lowp else 'fp32'} operands  "
                              f"{'col-tiled' if col_tiled else 'per-row  '}: fwd {t_f:7.2f} ms  bwd {t_b:7.2f} ms  (max rel err vs first {err:.1e})", flush=True)
            del q, k, v, gate, dacc, dl, m0
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
