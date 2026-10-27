"""Per-(row, sub-block) overhead of the Triton kernels: fixed number of rows, varying edges per row (a streamed
column sub-block has ~E/(P n_sub) edges spread over N/P_d rows, i.e. 1-3 edges per row).  Reports ms per pass and
the equivalent cost per row and per edge, for the forward partial and the backward partial (row pass + column pass),
plus the row-sized bookkeeping ops of the streamed layer (accumulate into the packed buffer)."""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from gt3d.attention import triton_kernel as old, triton_tiled as new  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402


def timeit(fn, iters=5):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3_500_000)
    ap.add_argument("--cols", type=int, default=1_700_000)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--edges", type=int, nargs="+", default=[875_000, 1_750_000, 3_500_000, 7_000_000, 14_000_000, 28_000_000, 56_000_000])
    ap.add_argument("--block_r", type=int, nargs="+", default=[16, 32])
    args = ap.parse_args()
    dev = torch.device("cuda")
    R, C, h, d = args.rows, args.cols, args.heads, args.dim
    gen = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(R, h, d, device=dev)
    k = torch.randn(C, h, d, device=dev)
    v = torch.randn(C, h, d, device=dev)
    gate = torch.randn(2, h, d, device=dev)
    dacc = torch.randn(R, h, d, device=dev)
    dl = torch.randn(R, h, device=dev)
    m0 = torch.zeros(R, h, device=dev)
    print(f"rows {R} cols {C} heads {h} dim {d} (h*d = {h*d})", flush=True)
    packed = torch.zeros(R, h * d + h + 64, device=dev)
    acc_like = torch.randn(R, h, d, device=dev)
    t_add = timeit(lambda: packed[:, :h * d].add_(acc_like.view(R, h * d)))
    print(f"packed[:, :hd] += acc   : {t_add:.2f} ms  ({t_add*1e6/R:.1f} ns/row)", flush=True)
    for E in args.edges:
        src = torch.randint(0, C, (E,), device=dev, generator=gen)
        dst = torch.randint(0, R, (E,), device=dev, generator=gen)
        et = torch.randint(0, 2, (E,), device=dev, generator=gen)
        crow, col, etype, _ = build_csr(torch.stack([src, dst]), R, et)
        col, etype = col.to(torch.int32), etype.to(torch.int32)
        nonempty = int(((crow[1:] - crow[:-1]) > 0).sum())
        t_f = timeit(lambda: old.attn_fwd_partial(q, k, v, crow, col, etype, gate, None, score="gated", norm="clampexp"))
        t_b = timeit(lambda: old.attn_bwd_partial(q, k, v, crow, col, etype, gate, None, m0, dacc, dl, score="gated", norm="clampexp"))
        print(f"E={E/1e6:5.1f}M ({E/R:4.1f} edges/row, {100*nonempty/R:4.1f}% rows non-empty): per-row  fwd {t_f:7.2f} ms "
              f"({t_f*1e6/R:5.1f} ns/row, {t_f*1e6/E:5.2f} ns/edge)   bwd {t_b:7.2f} ms ({t_b*1e6/R:5.1f} ns/row, {t_b*1e6/E:5.2f} ns/edge)",
              flush=True)
        for br in args.block_r:
            t_f = timeit(lambda: new.attn_fwd_partial(q, k, v, crow, col, etype, gate, None, score="gated", norm="clampexp", block_r=br))
            t_b = timeit(lambda: new.attn_bwd_partial(q, k, v, crow, col, etype, gate, None, m0, dacc, dl, score="gated", norm="clampexp", block_r=br))
            print(f"{'':>58}tiled BR={br:<3d} fwd {t_f:7.2f} ms ({t_f*1e6/R:5.1f} ns/row, {t_f*1e6/E:5.2f} ns/edge)   "
                  f"bwd {t_b:7.2f} ms ({t_b*1e6/R:5.1f} ns/row, {t_b*1e6/E:5.2f} ns/edge)", flush=True)
            # compact-accumulate (DCSR tiles over the non-empty rows, partials added in place: the streamed layer's
            # per-sub-block cost is this alone, vs tiled + the packed add above)
            packed.zero_()
            dqb = torch.zeros(R, h, d, device=dev)
            t_f = timeit(lambda: new.attn_fwd_partial(q, k, v, crow, col, etype, gate, None, score="gated", norm="clampexp", block_r=br,
                                                      accum=(packed[:, :h * d], packed[:, h * d:h * d + h])))
            t_b = timeit(lambda: new.attn_bwd_partial(q, k, v, crow, col, etype, gate, None, m0, dacc, dl, score="gated", norm="clampexp",
                                                      block_r=br, dq_accum=dqb))
            print(f"{'':>58}compact BR={br:<3d} fwd {t_f:7.2f} ms ({t_f*1e6/R:5.1f} ns/row, {t_f*1e6/E:5.2f} ns/edge)   "
                  f"bwd {t_b:7.2f} ms ({t_b*1e6/R:5.1f} ns/row, {t_b*1e6/E:5.2f} ns/edge)", flush=True)
            del dqb
        del src, dst, et, crow, col, etype
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
