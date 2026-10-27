"""Row-tiled kernels (triton_tiled) vs the per-row kernels (triton_kernel) on random graphs with hub rows,
empty rows and a row count that is not a multiple of the tile: forward partials, backward partials and the
normalised backward, for dot / gated scores and clamp-exp / softmax normalisation."""
import sys

import torch

sys.path.insert(0, ".")
from gt3d.attention import triton_kernel as old, triton_tiled as new  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402


def graph(n_rows, n_cols, n_edges, hubs, dev, gen):
    src = torch.randint(0, n_cols, (n_edges,), device=dev, generator=gen)
    dst = torch.randint(0, n_rows, (n_edges,), device=dev, generator=gen)
    for r, deg in hubs:                                                     # hub rows -> multi-segment tiles
        src = torch.cat([src, torch.randint(0, n_cols, (deg,), device=dev, generator=gen)])
        dst = torch.cat([dst, torch.full((deg,), r, device=dev)])
    et = torch.randint(0, 2, (src.numel(),), device=dev, generator=gen)
    crow, col, etype, _ = build_csr(torch.stack([src, dst]), n_rows, et)
    return crow, col.to(torch.int32), etype.to(torch.int32)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    ok = True
    for (n_rows, n_cols, n_edges, hubs) in [(1000, 700, 3000, [(5, 5000), (999, 3000)]), (4099, 2048, 8000, []),
                                            (50000, 30000, 1_000_000, [(17, 100000)])]:
        crow, col, etype = graph(n_rows, n_cols, n_edges, hubs, dev, gen)
        for h, d in [(2, 40), (4, 16), (8, 32)]:
            q = torch.randn(n_rows, h, d, device=dev, generator=gen)
            k = torch.randn(n_cols, h, d, device=dev, generator=gen)
            v = torch.randn(n_cols, h, d, device=dev, generator=gen)
            gate = torch.randn(2, h, d, device=dev, generator=gen)
            dacc = torch.randn(n_rows, h, d, device=dev, generator=gen)
            dl = torch.randn(n_rows, h, device=dev, generator=gen)
            do = torch.randn(n_rows, h, d, device=dev, generator=gen)
            for score in ("dot", "gated"):
                g = gate if score == "gated" else None
                for norm in ("clampexp", "softmax"):
                    a0, l0, m0 = old.attn_fwd_partial(q, k, v, crow, col, etype, g, None, score=score, norm=norm)
                    for br in (16, 32):
                        a1, l1, m1 = new.attn_fwd_partial(q, k, v, crow, col, etype, g, None, score=score, norm=norm,
                                                          block_r=br, seg_edges=512)
                        # compare normalised partials (m may differ by construction: compare acc*e^m and l*e^m consistently)
                        if norm == "softmax":
                            w0, w1 = torch.exp(m0), torch.exp(m1)
                            e_a = rel(a1 * w1.unsqueeze(-1), a0 * w0.unsqueeze(-1)); e_l = rel(l1 * w1, l0 * w0)
                        else:
                            e_a = rel(a1, a0); e_l = rel(l1, l0)
                        good = e_a < 1e-4 and e_l < 1e-4
                        ok &= good
                        print(f"fwd n={n_rows} h={h} d={d} {score}/{norm} BR={br}: acc {e_a:.1e} l {e_l:.1e} {'ok' if good else 'FAIL'}")
                    # backward of the partials
                    mm = m0
                    g0 = old.attn_bwd_partial(q, k, v, crow, col, etype, g, None, mm, dacc, dl, score=score, norm=norm)
                    g1 = new.attn_bwd_partial(q, k, v, crow, col, etype, g, None, mm, dacc, dl, score=score, norm=norm, seg_edges=512)
                    errs = [rel(x, y) for x, y in zip(g1[:3], g0[:3])] + ([rel(g1[3], g0[3])] if score == "gated" else [])
                    good = all(e < 1e-4 for e in errs)
                    ok &= good
                    print(f"bwd-partial n={n_rows} h={h} d={d} {score}/{norm}: dq/dk/dv{'/dgate' if score == 'gated' else ''} "
                          + " ".join(f"{e:.1e}" for e in errs) + (" ok" if good else " FAIL"))
                    if norm == "clampexp":
                        # compact-accumulate mode (streamed layer): only the rows with edges are tiled, partials added in
                        # place to row-strided buffers (BR=16 only: one more kernel variant to compile)
                        buf = torch.randn(n_rows, h * d + h + 7, device=dev, generator=gen)
                        ref_a = buf[:, :h * d] + a0.view(n_rows, h * d)
                        ref_l = buf[:, h * d:h * d + h] + l0
                        new.attn_fwd_partial(q, k, v, crow, col, etype, g, None, score=score, norm=norm, block_r=16,
                                             seg_edges=512, accum=(buf[:, :h * d], buf[:, h * d:h * d + h]))
                        e_a, e_l = rel(buf[:, :h * d], ref_a), rel(buf[:, h * d:h * d + h], ref_l)
                        good = e_a < 1e-4 and e_l < 1e-4
                        ok &= good
                        print(f"fwd-accum   n={n_rows} h={h} d={d} {score}: acc {e_a:.1e} l {e_l:.1e} {'ok' if good else 'FAIL'}")
                        dqb = torch.randn(n_rows, h, d, device=dev, generator=gen)
                        ref_dq = dqb + g0[0]
                        gc = new.attn_bwd_partial(q, k, v, crow, col, etype, g, None, mm, dacc, dl, score=score, norm=norm,
                                                  seg_edges=512, dq_accum=dqb)
                        assert gc[0] is None
                        errs = [rel(dqb, ref_dq), rel(gc[1], g0[1]), rel(gc[2], g0[2])] + ([rel(gc[3], g0[3])] if score == "gated" else [])
                        good = all(e < 1e-4 for e in errs)
                        ok &= good
                        print(f"bwd-accum   n={n_rows} h={h} d={d} {score}: dq/dk/dv{'/dgate' if score == 'gated' else ''} "
                              + " ".join(f"{e:.1e}" for e in errs) + (" ok" if good else " FAIL"))
                    # normalised backward
                    o = old.normalize(a0, l0, norm)
                    g0 = old.attn_bwd(q, k, v, crow, col, etype, g, None, o, l0, m0, do, score=score, norm=norm)
                    g1 = new.attn_bwd(q, k, v, crow, col, etype, g, None, o, l0, m0, do, score=score, norm=norm, seg_edges=512)
                    errs = [rel(x, y) for x, y in zip(g1[:3], g0[:3])] + ([rel(g1[3], g0[3])] if score == "gated" else [])
                    good = all(e < 1e-4 for e in errs)
                    ok &= good
                    print(f"bwd n={n_rows} h={h} d={d} {score}/{norm}: " + " ".join(f"{e:.1e}" for e in errs) + (" ok" if good else " FAIL"))
    print("[tiled] ALL PASSED" if ok else "[tiled] FAILED")


if __name__ == "__main__":
    main()
