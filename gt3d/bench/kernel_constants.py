"""Kernel cost constants for the white-box model (single GPU; companion of bench/hw_constants.py).

The attention and SpMM kernels are not pure streams: a per-row (forward, backward row pass) and a per-column
(backward column pass over the CSC) fixed cost sits next to the per-edge cost.  This script measures the three
constants of   T = c_e * E + c_r * rows + c_c * cols   per kernel, head configuration and regime by varying ONE
dimension at a time on synthetic uniform-random CSR blocks (slopes of three points each):

  per-row regime  : the kernels of the fused layer / dense blocks (>= 3.5 edges per row)
  compact regime  : the DCSR compact-accumulate tiled kernels of the streamed layer (hypersparse sub-blocks);
                    the row term counts the NON-EMPTY rows (the rows the compact kernels tile)
  spmm            : cuSPARSE CSR SpMM (GCN branch) A x and A^T g, dense rows regime and DCSR compact regime

plus the launch / autograd floors: CPU cost per forward op and per backward node of a chain of small ops.

    srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py
Prints the measurements and a `KERNEL = {...}` block for grid_select.py.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk, triton_tiled as tt  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402


def timeit(fn, iters=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def block(R, C, E, gen, dev):
    src = torch.randint(0, C, (E,), device=dev, generator=gen)
    dst = torch.randint(0, R, (E,), device=dev, generator=gen)
    et = torch.randint(0, 2, (E,), device=dev, generator=gen)
    crow, col, etype, _ = build_csr(torch.stack([src, dst]), R, et)
    nonempty = int(((crow[1:] - crow[:-1]) > 0).sum())
    return crow.to(torch.int32).contiguous(), col.to(torch.int32), etype.to(torch.int32), crow, col, nonempty


def slope(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


LOWP = False


def measure(regime, h, dh, dev, gen, say):
    """Returns dict of ns constants for fwd / bwd attention and spmm fwd / transpose."""
    d = h * dh
    if regime == "per_row":
        base = dict(R=612_352, C=612_352, E=8_837_500)
        var = dict(E=[4_400_000, 8_837_500, 17_675_000], R=[153_088, 612_352, 1_224_704], C=[153_088, 612_352, 1_224_704])
    else:
        base = dict(R=1_750_000, C=400_000, E=1_000_000)          # ~0.57 edges per row (papers100M 16x4 n_sub 16)
        var = dict(E=[500_000, 1_000_000, 2_000_000], R=[875_000, 1_750_000, 3_500_000], C=[100_000, 400_000, 1_600_000])
    gate = torch.randn(2, h, dh, device=dev)
    res = {}
    for dim, values in var.items():
        xs, t_f, t_b, t_s, t_st = [], [], [], [], []
        for val in values:
            R, C, E = base["R"], base["C"], base["E"]
            if dim == "R": R = val
            elif dim == "C": C = val
            else: E = val
            crow32, col32, et32, crow, col, nonempty = block(R, C, E, gen, dev)
            q = torch.randn(R, h, dh, device=dev); k = torch.randn(C, h, dh, device=dev); v = torch.randn(C, h, dh, device=dev)
            dacc = torch.randn(R, h, dh, device=dev); dl = torch.randn(R, h, device=dev); m0 = torch.zeros(R, h, device=dev)
            if LOWP:
                q, k, v, dacc = (t.to(torch.bfloat16) for t in (q, k, v, dacc))
            if regime == "per_row":
                f = lambda: tk._attn_fwd_partial_rows(q, k, v, crow32, col32, et32, gate, None, score="gated", norm="clampexp", seg_edges=512)
                b = lambda: tk._attn_bwd_partial_rows(q, k, v, crow32, col32, et32, gate, None, m0, dacc, dl, score="gated", norm="clampexp", seg_edges=512)
                xval = {"E": E, "R": R, "C": C}[dim]
            else:
                packed = torch.zeros(R, h * dh + h, device=dev); dqb = torch.zeros(R, h, dh, device=dev)
                f = lambda: tt.attn_fwd_partial(q, k, v, crow32, col32, et32, gate, None, score="gated", norm="clampexp", accum=(packed[:, :h * dh], packed[:, h * dh:]))
                b = lambda: tt.attn_bwd_partial(q, k, v, crow32, col32, et32, gate, None, m0, dacc, dl, score="gated", norm="clampexp", dq_accum=dqb)
                xval = {"E": E, "R": nonempty, "C": C}[dim]          # compact: rows = non-empty rows
            tf, tb = timeit(f), timeit(b)
            # SpMM (GCN): dense-rows regime uses the full CSR, compact regime the DCSR of the non-empty rows
            w = torch.rand(E, device=dev)
            if regime == "per_row":
                a = torch.sparse_csr_tensor(crow32, col32, w, size=(R, C)); out_rows = R
            else:
                deg = crow[1:] - crow[:-1]; rows_c = torch.nonzero(deg > 0).squeeze(1)
                crow_c = torch.zeros(rows_c.numel() + 1, dtype=torch.long, device=dev); crow_c[1:] = torch.cumsum(deg[rows_c], 0)
                a = torch.sparse_csr_tensor(crow_c.to(torch.int32), col32, w, size=(rows_c.numel(), C)); out_rows = rows_c.numel()
            perm_t = torch.argsort(col, stable=True)
            ccol = torch.zeros(C + 1, dtype=torch.long, device=dev); ccol[1:] = torch.cumsum(torch.bincount(col, minlength=C), 0)
            rows = torch.repeat_interleave(torch.arange(R, device=dev), crow[1:] - crow[:-1])
            at = torch.sparse_csr_tensor(ccol.to(torch.int32), rows[perm_t].to(torch.int32), w[perm_t], size=(C, R))
            xc = torch.randn(C, d, device=dev); gr = torch.randn(R, d, device=dev)
            ts = timeit(lambda: torch.sparse.mm(a, xc)); tst = timeit(lambda: torch.sparse.mm(at, gr))
            xs.append(xval); t_f.append(tf); t_b.append(tb); t_s.append(ts); t_st.append(tst)
            if regime != "per_row":
                del packed, dqb
            say(f"    {regime} h{h} d{dh} vary {dim}={val:>9d} (E {E/1e6:5.1f}M, R {R/1e6:5.2f}M, C {C/1e6:5.2f}M, nonempty {nonempty/1e6:5.2f}M): "
                f"attn fwd {tf:7.2f} bwd {tb:7.2f} | spmm {ts:6.2f} spmm^T {tst:6.2f} ms")
            del q, k, v, dacc, dl, m0, w, a, at, xc, gr, crow32, col32, et32, crow, col
            torch.cuda.empty_cache()
        res[f"attn_fwd_{dim}"] = slope(xs, t_f) * 1e6; res[f"attn_bwd_{dim}"] = slope(xs, t_b) * 1e6
        res[f"spmm_{dim}"] = slope(xs, t_s) * 1e6; res[f"spmmT_{dim}"] = slope(xs, t_st) * 1e6
    return res


def floors(dev, say):
    x = torch.randn(3648, 128, device=dev, requires_grad=True)
    n_ops = 300

    def chain():
        y = x
        for _ in range(n_ops):
            y = y * 1.01 + 0.5
        return y.sum()
    t_f = timeit(chain, iters=5)
    def fb():
        chain().backward()
    t_fb = timeit(fb, iters=5)
    say(f"    chain of {n_ops} small ops on [3648,128]: forward {t_f:.2f} ms ({t_f*1e3/n_ops/2:.1f} us per op), "
        f"forward+backward {t_fb:.2f} ms ({(t_fb-t_f)*1e3/n_ops/2:.1f} us per backward node)")
    return dict(fwd_op_us=t_f * 1e3 / n_ops / 2, bwd_node_us=(t_fb - t_f) * 1e3 / n_ops / 2)


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    say = lambda *a: print(*a, flush=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="2x40,4x32,8x32", help="HxD_head list; head-slice configurations of the head grids: 1x32,2x32,1x40")
    ap.add_argument("--regimes", default="per_row,compact")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lowp", action="store_true", help="bf16 q / k / v / dacc operands (fp32 accumulation): the --kernel_bf16 regime")
    args = ap.parse_args()
    global LOWP
    LOWP = args.lowp
    cfgs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]
    out = {"floors": floors(dev, say), "attn": {}}
    for regime in args.regimes.split(","):
        for h, dh in cfgs:
            say(f"[kernel] {regime} h={h} d_head={dh}")
            r = measure(regime, h, dh, dev, gen, say)
            out["attn"][f"{regime}_h{h}d{dh}"] = r
            say(f"[kernel]   ns: attn fwd  per edge {r['attn_fwd_E']:6.2f}  per row {r['attn_fwd_R']:6.1f}  per col {r['attn_fwd_C']:6.1f} | "
                f"attn bwd per edge {r['attn_bwd_E']:6.2f}  per row {r['attn_bwd_R']:6.1f}  per col {r['attn_bwd_C']:6.1f}")
            say(f"[kernel]   ns: spmm A x per edge {r['spmm_E']:6.2f} per row {r['spmm_R']:6.1f} per col {r['spmm_C']:6.1f} | "
                f"spmm A^T g per edge {r['spmmT_E']:6.2f} per row {r['spmmT_R']:6.1f} per col {r['spmmT_C']:6.1f}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"kernel_constants{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    say(f"[kernel] written {path}")


if __name__ == "__main__":
    main()
