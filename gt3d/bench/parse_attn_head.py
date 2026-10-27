"""Tabulate the ATTN lines of the head-parallel study logs (scripts/run_attn_head.sh) as markdown.

    python bench/parse_attn_head.py logs/attn_head_*.log

One row per (GPUs, graph/model), one column per strategy + grid: "fwd + bwd = total (mem GB)".
"""
import re
import sys
from collections import OrderedDict

pat = re.compile(r"ATTN strategy=(\S+) grid=(\S+) gpus=(\d+) data=(\S+) hidden=(\d+) heads=(\d+) .* fwd_ms=([\d.]+) bwd_ms=([\d.]+) "
                 r"layer_ms=([\d.]+) .*peak_extra_gb=([\d.]+)")
rows = OrderedDict()
for path in sys.argv[1:]:
    for line in open(path):
        m = pat.search(line)
        if not m:
            continue
        strat, grid, gpus, data, hid, heads, fwd, bwd, tot, mem = m.groups()
        key = (int(gpus), data, int(hid), int(heads))
        label = f"{strat} {grid}"
        rows.setdefault(key, OrderedDict())[label] = (float(fwd), float(bwd), float(tot), float(mem))
for key in sorted(rows):
    gpus, data, hid, heads = key
    print(f"\n**{gpus} GPUs, {data} hidden {hid} / {heads} heads** (ms per layer fwd + bwd = total, peak GB above inputs)\n")
    print("| strategy / grid (P_d x P_o x P_i) | fwd | bwd | total | mem |")
    print("|---|---|---|---|---|")
    best = min(v[2] for lab, v in rows[key].items() if "nocomm" not in lab)      # compute-only rows are not candidates
    for label, (fwd, bwd, tot, mem) in rows[key].items():
        mark = "**" if tot == best else ""
        print(f"| {label} | {fwd:.0f} | {bwd:.0f} | {mark}{tot:.0f}{mark} | {mem:.1f} |")
