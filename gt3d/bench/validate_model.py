"""Validation of the white-box model (gt3d/dist/grid_select.py) against the measured points of
bench/model_validation_data.py: prediction vs measurement, per-column ranking (does the model pick the measured-best
grid?), and the breakdown of the residual.  Nothing here feeds back into the model.

    python bench/validate_model.py [--sections fused,attn,streamed,memory]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist import grid_select as gs  # noqa: E402
from bench.model_validation_data import (GRAPHS, HEADS, FUSED_IEEE, FUSED_TF32, ATTN_ONLY, ATTN_SPLIT, STREAMED,  # noqa: E402
                                         STREAMED_TF32, PAIRS_NONEMPTY)


def graph(name):
    g = GRAPHS[name]
    return gs.Graph(g["n"], g["e_attn"], g["e_gcn"], g["f_in"], name=name)     # name -> per-graph locality coefficients


def section_fused():
    print("=" * 100)
    print("(a) full model, fused 2D layer: ms per epoch, model vs measured; per GPU count the measured best grid and the model's pick")
    tot_err, tot_cols, hits, loss = [], 0, 0, []
    from bench.model_validation_data import FUSED_SYNC
    for tf32, table in ((False, FUSED_IEEE), (True, FUSED_TF32), (True, FUSED_SYNC)):
        gs.ONE_CROSS_NODE_AXIS = table is not FUSED_SYNC        # the pre-sync tables carry the two-cross-node-group penalty
        for (gname, hidden), rows in table.items():
            g, m = graph(gname), gs.Model(hidden, HEADS[hidden], 4, tf32)
            byP = {}
            for pd, po, ms, _ in rows:
                byP.setdefault(pd * po, []).append((pd, po, ms, gs.predict_fused_ms(pd, po, g, m)))
            print(f"--- {gname} hidden {hidden} ({'TF32' if tf32 else 'IEEE'} matmuls{', default chain + sync' if table is FUSED_SYNC else ''})")
            for P in sorted(byP):
                pts = byP[P]
                adm = [r for r in pts if gs.admissible(r[0], r[1])] or pts
                best_m = min(pts, key=lambda r: r[2]); best_p = min(adm, key=lambda r: r[3])
                if len(pts) > 1:
                    tot_cols += 1; hits += best_m[:2] == best_p[:2]; loss.append(best_p[2] / best_m[2])
                line = "  ".join(f"{pd}x{po}{'' if gs.admissible(pd, po) else 'x'}: {ms:.0f}/{pr:.0f} ({100*(pr-ms)/ms:+.0f}%)" for pd, po, ms, pr in sorted(pts, key=lambda r: r[2]))
                flag = "" if len(pts) == 1 else ("  pick OK" if best_m[:2] == best_p[:2] else f"  pick {best_p[0]}x{best_p[1]} = {best_p[2]/best_m[2]:.2f}x of best")
                print(f"  P={P:3d}: {line}{flag}")
                tot_err += [abs(pr - ms) / ms for pd, po, ms, pr in pts if gs.admissible(pd, po)]
    print(f"  => mean |error| over admissible grids {100*np.mean(tot_err):.0f} %, measured-best grid picked in {hits}/{tot_cols} columns, "
          f"mean cost of a wrong pick {100*(np.mean(loss)-1):.1f} %, worst {100*(max(loss)-1):.1f} %")
    # breakdown of one point per scale
    for gname, hidden, pd, po, tf32 in (("products", 128, 2, 2, False), ("products", 128, 4, 4, False), ("products", 128, 16, 4, True), ("reddit", 128, 4, 4, True), ("reddit", 128, 16, 4, True)):
        tot, parts = gs.predict_fused_ms(pd, po, graph(gname), gs.Model(hidden, HEADS[hidden], 4, tf32), breakdown=True)
        print(f"  breakdown {gname} h{hidden} {pd}x{po}: total {tot:.0f} ms = " + ", ".join(f"{k} {v:.0f}" for k, v in parts.items()))


def section_attn():
    print("=" * 100)
    print("(b) attention operator only (one layer fwd+bwd, ms): fused family vs head grids; model vs measured, best shape per GPU count")
    errs, hits, cols, loss = [], 0, 0, []
    gs.ONE_CROSS_NODE_AXIS = True                                 # measured before the pre-collective sync existed
    for (gname, hidden), byP in ATTN_ONLY.items():
        g, m = graph(gname), gs.Model(hidden, HEADS[hidden], 4, False)
        print(f"--- {gname} hidden {hidden} / {HEADS[hidden]} heads")
        for P, rows in sorted(byP.items()):
            pts = [(pd, po, pi, fam, ms, gs.attn_operator_ms(pd, po, pi, g, m, fam)) for pd, po, pi, fam, ms in rows]
            adm = [r for r in pts if gs.admissible(r[0], r[1], r[2])] or pts
            best_m = min(pts, key=lambda r: r[4]); best_p = min(adm, key=lambda r: r[5])
            cols += 1; hits += best_m[:4] == best_p[:4]; loss.append(best_p[4] / best_m[4])
            errs += [abs(pr - ms) / ms for pd, po, pi, fam, ms, pr in pts if gs.admissible(pd, po, pi)]
            line = "  ".join(f"{'F' if fam == 'fused' else 'H'}({pd},{po},{pi}){'' if gs.admissible(pd, po, pi) else 'x'} {ms:.0f}/{pr:.0f}" for pd, po, pi, fam, ms, pr in sorted(pts, key=lambda r: r[4]))
            flag = "pick OK" if best_m[:4] == best_p[:4] else f"pick {best_p[3][0].upper()}{best_p[:3]} = {best_p[4]/best_m[4]:.2f}x"
            print(f"  P={P:3d}: {line}   [{flag}]")
    print(f"  => mean |error| over admissible shapes {100*np.mean(errs):.0f} %, best shape picked in {hits}/{cols} columns, mean cost of a wrong pick {100*(np.mean(loss)-1):.1f} %, worst {100*(max(loss)-1):.1f} %")
    print("  compute-only / exposed-communication split (measured compute, exposed | model compute, comm):")
    for (gname, hidden, P, pd, po, pi, fam), (comp, exp) in ATTN_SPLIT.items():
        g, m = graph(gname), gs.Model(hidden, HEADS[hidden], 4, False)
        tot = gs.attn_operator_ms(pd, po, pi, g, m, fam)
        saved = gs.HW["nccl"]
        gs.HW["nccl"] = {k: {kk: (0.0, 1e12) for kk in v} for k, v in saved.items()}     # free communication
        comp_model = gs.attn_operator_ms(pd, po, pi, g, m, fam)
        gs.HW["nccl"] = saved
        print(f"    {gname:8s} h{hidden:<3d} P={P:3d} {fam:5s} ({pd},{po},{pi}): measured {comp:6.1f} + {exp:6.1f} | model {comp_model:6.1f} + {tot-comp_model:6.1f}")


def section_sync():
    gs.ONE_CROSS_NODE_AXIS = False
    print("=" * 100)
    print("(e) attention operator at 16 GPUs with the pre-collective sync (the shapes the one-cross-node-axis rule excluded): measured / model")
    from bench.model_validation_data import ATTN_ONLY_SYNC, ATTN_ONLY_SYNC_64, FUSED_SYNC_64
    errs, hits, cols = [], 0, 0
    for table in (ATTN_ONLY_SYNC, ATTN_ONLY_SYNC_64):
        for (gname, hidden), byP in table.items():
            g, m = graph(gname), gs.Model(hidden, HEADS[hidden], 4, False)
            for P, rows in byP.items():
                pts = [(pd, po, pi, fam, ms, gs.attn_operator_ms(pd, po, pi, g, m, fam)) for pd, po, pi, fam, ms in rows]
                best_m = min(pts, key=lambda r: r[4]); best_p = min(pts, key=lambda r: r[5])
                cols += 1; hits += best_m[:4] == best_p[:4]
                for pd, po, pi, fam, ms, pr in pts:
                    errs.append(abs(pr - ms) / ms)
                    print(f"  {gname} h{hidden} P={P} {fam[0].upper()}({pd},{po},{pi}): measured {ms:6.1f}  model {pr:6.1f} ({100*(pr-ms)/ms:+.0f}%)")
                print(f"    -> measured best {best_m[3][0].upper()}{best_m[:3]} {best_m[4]:.1f}, model picks {best_p[3][0].upper()}{best_p[:3]} ({best_p[4]/best_m[4]:.2f}x of best)")
    print(f"  => attention operator after the fix: mean |error| {100*np.mean(errs):.0f} %, best shape picked in {hits}/{cols} columns")
    print("  full model, 64 GPUs with the sync (TF32): measured / model")
    for (gname, hidden), rows in FUSED_SYNC_64.items():
        g, m = graph(gname), gs.Model(hidden, HEADS[hidden], 4, True)
        for pd, po, ms, peak in rows:
            pr = gs.predict_fused_ms(pd, po, g, m)
            print(f"    {gname} h{hidden} {pd}x{po}: {ms:6.1f} / {pr:6.1f} ({100*(pr-ms)/ms:+.0f}%)")


def section_head():
    """Full model on the head grid (predict_head_ms) vs the fused 2D layer (predict_fused_ms): attention-only (HEAD_MODEL)
    and GCN + attention (HEAD_MODEL_GCN)."""
    from bench.model_validation_data import HEAD_MODEL, HEAD_MODEL_GCN
    for label, table, with_gcn in (("attention-only", HEAD_MODEL, False), ("GCN + attention", HEAD_MODEL_GCN, True)):
        print("=" * 100)
        print(f"(h) {label} full model: head grid vs fused 2D, measured / model, and the memory model of the head layer")
        errs, hits, cols = [], 0, 0
        for (gname, hidden), byP in table.items():
            g = graph(gname)
            m = gs.Model(hidden, HEADS[hidden], 4, True, with_gcn=with_gcn)
            for P, rows in byP.items():
                pts = []
                for pd, po, pi, fam, ms, peak in rows:
                    if fam == "head":
                        pr = gs.predict_head_ms(pd, po, pi, g, m)
                        mem = gs.peak_memory_head_bytes(pd, po, pi, g.n, g.e_attn, hidden, g.f_in, 4, HEADS[hidden],
                                                        with_gcn=with_gcn, e_gcn=g.e_gcn) / gs.GB
                    else:
                        pr = gs.predict_fused_ms(pd, po, g, m)
                        mem = gs.peak_memory_bytes(pd, po, g.n, g.e_attn, g.e_gcn if with_gcn else 0, hidden, g.f_in, 4, heads=HEADS[hidden]) / gs.GB
                    pts.append((pd, po, pi, fam, ms, pr, peak, mem))
                best_m = min(pts, key=lambda r: r[4]); best_p = min(pts, key=lambda r: r[5])
                cols += 1; hits += best_m[:4] == best_p[:4]
                for pd, po, pi, fam, ms, pr, peak, mem in pts:
                    errs.append(abs(pr - ms) / ms)
                    pk = f"  peak {peak:5.1f} / {mem:5.1f} GB" if peak else ""
                    print(f"  {gname} h{hidden} P={P} {fam[0].upper()}({pd},{po},{pi}): measured {ms:7.1f}  model {pr:7.1f} ({100*(pr-ms)/ms:+.0f}%){pk}")
                print(f"    -> measured best {best_m[3][0].upper()}{best_m[:3]} {best_m[4]:.1f}, model picks {best_p[3][0].upper()}{best_p[:3]} ({best_p[4]/best_m[4]:.2f}x of best)")
        if errs:
            print(f"  => {label} model: mean |error| {100*np.mean(errs):.0f} %, best shape picked in {hits}/{cols} columns")


def section_streamed():
    print("=" * 100)
    print("(c) streamed layer, papers100M: ms per epoch, measured components vs model")
    g = graph("papers")
    for pd, po, hidden, n_sub, rowside, exposed, proj, attn, rs, fwd, sbwd, bwd, epoch, peak in STREAMED:
        m = gs.Model(hidden, HEADS[hidden], 4, False)
        o = gs.streamed_layer_ms(pd, po, n_sub, g, m)
        L = 4
        pred = gs.predict_streamed_ms(pd, po, g.n, g.e_attn, g.e_gcn, hidden, n_sub, HEADS[hidden], 4, g.f_in, False)
        print(f"  {pd:2d}x{po:<2d} h{hidden} n_sub {n_sub:<3d}: epoch measured {epoch:7.0f} model {pred:7.0f} ({100*(pred-epoch)/epoch:+.0f}%) | "
              f"rowside {rowside}/{L*o['rowside']:.0f} exposed {exposed}/{L*o['exposed']:.0f} proj {proj}/{L*(o['proj']+o['spmm']):.0f} "
              f"attn {attn}/{L*o['attn']:.0f} rs {rs}/{L*o['rs']:.0f} | fwd {fwd}/{L*o['fwd']:.0f} bwd {bwd}/{L*o['bwd']:.0f}")
        if peak:
            mem = gs.peak_memory_bytes(pd, po, g.n, g.e_attn, g.e_gcn, hidden, g.f_in, 4, n_sub, streamed=True, heads=HEADS[hidden]) / gs.GB
            print(f"           peak GB measured {peak:.1f} model {mem:.1f}")
    for pd, po, hidden, n_sub, epoch in STREAMED_TF32:
        pred = gs.predict_streamed_ms(pd, po, g.n, g.e_attn, g.e_gcn, hidden, n_sub, HEADS[hidden], 4, g.f_in, True)
        print(f"  {pd:2d}x{po:<2d} h{hidden} n_sub {n_sub:<3d} TF32: measured {epoch:7.0f} model {pred:7.0f} ({100*(pred-epoch)/epoch:+.0f}%)")
    print("  non-empty (row, sub-block) fraction, measured vs negative-binomial(kappa=0.8) vs Poisson:")
    for lam, f in PAIRS_NONEMPTY:
        print(f"    lam {lam:.2f}: measured {100*f:.1f} %  NB {100*gs.nonempty_fraction(lam, 0.8):.1f} %  Poisson {100*gs.nonempty_fraction(lam, None):.1f} %")
    best, ranked = gs.select_grid_streamed(64, g.n, g.e_attn, g.e_gcn, 64, g.f_in)
    print("  64 GPUs hidden 64 ranking (streamed, memory-feasible): " + ", ".join(f"{pd}x{po} n{ns} {ms:.0f} ms/{mem:.0f} GB" for pd, po, ns, ms, mem in ranked[:6]))


def section_memory():
    print("=" * 100)
    print("(d) peak memory, fused layer without checkpointing: measured vs counted model (GB)")
    errs = []
    for (gname, hidden), rows in FUSED_IEEE.items():
        g = GRAPHS[gname]
        line = []
        for pd, po, _, peak in rows:
            if peak is None:
                continue
            pred = gs.peak_memory_bytes(pd, po, g["n"], g["e_attn"], g["e_gcn"], hidden, g["f_in"], 4, heads=HEADS[hidden]) / gs.GB
            errs.append(abs(pred - peak) / peak)
            line.append(f"{pd}x{po} {peak:.1f}/{pred:.1f}")
        print(f"  {gname} h{hidden}: " + "  ".join(line))
    print(f"  => mean |error| {100*np.mean(errs):.0f} %, max {100*np.max(errs):.0f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="fused,attn,sync,head,streamed,memory")
    args = ap.parse_args()
    for s in args.sections.split(","):
        globals()[f"section_{s}"]()
