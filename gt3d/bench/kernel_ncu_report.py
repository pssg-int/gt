"""Join the Nsight Compute report of bench/kernel_ncu.py (bench/kernel_ncu.csv, --page details) with the launch order
and print, per (configuration, pass, edges per row): duration, measured DRAM bytes against the bytes of the analytic model
(beta = 8 H d_h (d + c) per program), DRAM throughput, compute throughput, registers, achieved occupancy, and L2 atomics."""
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(l for l in open(os.path.join(HERE, "kernel_ncu.csv")) if not l.startswith("==")))
order = json.load(open(os.path.join(HERE, "kernel_ncu_order.json")))
by_id = defaultdict(dict)
for r in rows:
    by_id[int(r["ID"])][r["Metric Name"]] = (r["Metric Value"], r["Metric Unit"])
C = dict(fwd=1.0, row=1.5, col=2.0)


def val(m, *names):
    for n in names:
        if n in m:
            v, u = m[n]
            try:
                x = float(v.replace(",", ""))
            except ValueError:
                return None
            scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1e-9, "usecond": 1e-6,
                     "msecond": 1e-3, "second": 1, "us": 1e-6, "ms": 1e-3, "ns": 1e-9}.get(u, 1)
            return x * scale
    return None


if "--names" in sys.argv:
    print(sorted(by_id[0]))
    sys.exit()
print(f"{'case':24s} {'ms':>7s} {'DRAM GB':>8s} {'model GB':>8s} {'ratio':>6s} {'DRAM %':>7s} {'SM %':>6s} "
      f"{'regs':>5s} {'th.occ':>6s} {'occ %':>6s} {'cyc/iss':>7s} {'L2 hit':>6s} {'atom/red':>9s}")
for i, o in enumerate(order):
    m = by_id.get(i)
    if m is None:
        continue
    h, dh = map(int, o["cfg"].split("x"))
    n, d = o["n"], o["d"]
    model = 8 * h * dh * (d + C[o["pass_"]]) * n
    dram = (val(m, "dram__bytes_read.sum") or 0) + (val(m, "dram__bytes_write.sum") or 0)
    dur = val(m, "Duration", "gpu__time_duration.sum")
    dram_pct = val(m, "DRAM Throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    sm_pct = val(m, "Compute (SM) Throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    regs = val(m, "Registers Per Thread", "launch__registers_per_thread")
    occ = val(m, "Achieved Occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active")
    atom = (val(m, "lts__t_requests_srcunit_tex_op_red.sum") or 0) + (val(m, "lts__t_requests_srcunit_tex_op_atom.sum") or 0)
    tocc = val(m, "Theoretical Occupancy"); cpi = val(m, "Warp Cycles Per Issued Instruction"); l2 = val(m, "L2 Hit Rate")
    label = f"{o['score']} {o['cfg']} {o['pass_']} d={d}"
    print(f"{label:24s} {dur*1e3 if dur else float('nan'):7.3f} {dram/1e9:8.3f} {model/1e9:8.3f} {dram/model:6.2f} "
          f"{dram_pct if dram_pct is not None else float('nan'):7.1f} {sm_pct if sm_pct is not None else float('nan'):6.1f} "
          f"{regs if regs else float('nan'):5.0f} {tocc:6.1f} {occ if occ is not None else float('nan'):6.1f} {cpi:7.1f} {l2:6.1f} {atom:9.0f}")
