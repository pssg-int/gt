"""Cost of a gather served from L2, relative to HBM, for the three passes of the attention kernel (single GPU).

The forward pass and the backward row pass gather K / V rows of the COLUMN block, the backward column pass gathers
Q / dO rows of the ROW block.  A block whose gathered side fits in L2 (C x 2w or R x 2w <= ~32 MB) is timed against the
bytes it moves; time / (bytes / B) is the L2 factor lambda of that pass.

    srun ... bash scripts/run_on_node.sh python bench/kernel_l2.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.backend_torch import build_csr  # noqa: E402
from bench.kernel_theory import kernels, timeit, hbm_read_gbs  # noqa: E402


def block(R, C, E, gen, dev):
    src = torch.randint(0, C, (E,), device=dev, generator=gen)
    dst = torch.randint(0, R, (E,), device=dev, generator=gen)
    et = torch.randint(0, 2, (E,), device=dev, generator=gen)
    crow, col, etype, _ = build_csr(torch.stack([src, dst]), R, et)
    return crow.to(torch.int32).contiguous(), col.to(torch.int32).contiguous(), etype.to(torch.int32).contiguous()


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    say = lambda *a: print(*a, flush=True)  # noqa: E731
    B = hbm_read_gbs(dev) * 1e9
    out = dict(gpu=torch.cuda.get_device_name(), hbm_read_gbs=B / 1e9, cases=[])
    say(f"[l2] {out['gpu']} HBM read {B/1e9:.0f} GB/s")
    for cfg in ("4x32", "2x32", "1x32", "8x32"):
        h, dh = map(int, cfg.split("x"))
        w = 4 * min(h, 4) * dh
        # (R, C, E): gathered side resident (small C for fwd / row pass, small R for the column pass) and not resident
        for R, C, E in ((612352, 8192, 39_190_528), (612352, 612352, 39_190_528), (8192, 612352, 39_190_528)):
            gate = torch.randn(2, h, dh, device=dev)
            q = torch.randn(R, h, dh, device=dev); k = torch.randn(C, h, dh, device=dev); v = torch.randn(C, h, dh, device=dev)
            dacc = torch.randn(R, h, dh, device=dev); dl = torch.randn(R, h, device=dev); m0 = torch.zeros(R, h, device=dev)
            blk = block(R, C, E, gen, dev)
            ks = kernels(q, k, v, *blk, gate, m0, dacc, dl)
            for f in ks.values():                        # autotune on this block
                f()
            t = {name: timeit(f) for name, f in ks.items()}
            out["cases"].append(dict(cfg=cfg, R=R, C=C, E=E, **t))
            say(f"[l2] {cfg} R {R:7d} C {C:7d} E {E}: fwd {t['fwd']:8.3f} row {t['row']:8.3f} col {t['col']:8.3f} ms   "
                f"(K/V {C * 2 * w / 2**20:.0f} MB, Q/dO {R * 2 * w / 2**20:.0f} MB per head group)")
            del q, k, v, dacc, dl, m0, blk, ks
            torch.cuda.empty_cache()
            for mod_name in ("_attn_fwd_kernel", "_attn_bwd_kernel", "_attn_bwd_kv_kernel"):
                from gt3d.attention import triton_kernel as tk
                getattr(tk, mod_name).cache.clear()      # retune for the next block
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_l2.json")
    json.dump(out, open(path, "w"), indent=1)
    say(f"[l2] written {path}")


if __name__ == "__main__":
    main()
