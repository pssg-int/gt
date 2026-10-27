"""Tabulate the DAC'26 baseline sweep (scripts/run_dac26_strategies.sh) as markdown.

    python bench/parse_dac26.py logs/sbatch_dac26_*.log

Per part (attention-only operator: ms per layer fwd + bwd; full model: ms per epoch) one row per (GPUs, graph, model):
GP-AG (paper Alg. 1), GP-AG-x (x-gather variant), GP-A2A (paper Alg. 2), DAC best = the faster of the two paper
strategies (oracle in place of AGP, an upper bound on its choice), ours best (layout), --head_auto's choice, and the
speed-up of ours over DAC best.  Peak GB in parentheses; "OOM" when the run did not fit, "(ckpt)" when the full-model
run needed activation checkpointing.
"""
import re
import sys
from collections import OrderedDict

hdr = re.compile(r"^### (\S+) \| gpus=(\d+) (\S+) h(\d+)/(\d+) (.*?)\s+\d\d:\d\d:\d\d$")
retry = re.compile(r"^### (\S+) \| retry with --checkpoint")
attn = re.compile(r"^ATTN strategy=(\S+) grid=(\S+) .* layer_ms=([\d.]+) .*peak_extra_gb=([\d.]+)")
steady = re.compile(r"^STEADY .*grid=(\S+) .*fused=(\d) .*ckpt=(\d) ms_per_epoch=([\d.]+) peak_gb=([\d.]+)")

res = {"attn": OrderedDict(), "model": OrderedDict()}
for path in sys.argv[1:]:
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        m = hdr.match(line)
        if m:
            label, gpus, data, hid, heads, rest = m.groups()
            cur = (label, (int(gpus), data, int(hid), int(heads)), rest)
            continue
        if cur is None:
            continue
        label, key, rest = cur
        if retry.match(line):
            continue
        m = attn.match(line)
        if m:
            strat, grid, ms, gb = m.groups()
            res["attn"].setdefault(key, []).append((label, f"{strat} {grid}", float(ms), float(gb), False))
            continue
        m = steady.match(line)
        if m:
            grid, fused, ckpt, ms, gb = m.groups()
            lay = ("fused " if fused == "1" else "head ") + grid
            res["model"].setdefault(key, []).append((label, lay, float(ms), float(gb), ckpt == "1"))
            continue
        if "OutOfMemory" in line:
            part = "attn" if "--grid" not in rest else "model"
            res[part].setdefault(key, []).append((label, rest, None, None, False))


def cell(entries):
    ok = [e for e in entries if e[2] is not None]
    if not ok:
        return ("OOM" if entries else "–"), None
    e = min(ok, key=lambda t: t[2])
    return f"{e[2]:.1f} ({e[3]:.1f}){' ckpt' if e[4] else ''}", e


for part, unit in (("attn", "attention-only operator, ms per layer fwd + bwd (peak GB above inputs)"),
                   ("model", "full Exphormer model (GCN + attention), ms per epoch (peak GB)")):
    if not res[part]:
        continue
    print(f"\n### {part}: {unit}\n")
    print("| GPUs | graph | model | GP-AG | GP-AG-x | GP-A2A | DAC best | ours best | layout | head_auto | ours / DAC |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(res[part], key=lambda k: (k[1], k[2], k[0])):
        gpus, data, hid, heads = key
        rows = res[part][key]
        by = {lab: [r for r in rows if r[0] == lab] for lab in ("GP-AG", "GP-AG-x", "GP-A2A", "ours", "ours-auto")}
        c_ag, e_ag = cell(by["GP-AG"])
        c_agx, _ = cell(by["GP-AG-x"])
        c_a2a, e_a2a = cell(by["GP-A2A"]) if by["GP-A2A"] else ("n/a", None)
        dac = [e for e in (e_ag, e_a2a) if e is not None]
        dac_best = min(dac, key=lambda t: t[2]) if dac else None
        c_ours, e_ours = cell(by["ours"] + by["ours-auto"])
        c_auto, e_auto = cell(by["ours-auto"])
        auto = f"{e_auto[1]} {e_auto[2]:.1f}" if e_auto else "–"
        sp = f"**{dac_best[2] / e_ours[2]:.2f}x**" if dac_best and e_ours else "–"
        print(f"| {gpus} | {data.split('_')[0]} | h{hid}/{heads} | {c_ag} | {c_agx} | {c_a2a} | "
              f"{f'{dac_best[2]:.1f} ' + ('A2A' if dac_best is e_a2a else 'AG') if dac_best else '–'} | {c_ours} | "
              f"{e_ours[1] if e_ours else '–'} | {auto} | {sp} |")
