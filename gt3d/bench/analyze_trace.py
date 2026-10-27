"""Analyse a torch.profiler chrome trace of one training step (train_dist.py --profile with GT3D_PROF_TRACE=<prefix>):
GPU busy time vs wall time per phase, idle gaps, every NCCL kernel with its duration (waiting included), and the
CPU-side duration of the autograd backward.  Usage: python bench/analyze_trace.py <trace.json> [<trace.json> ...]
(The fwd / bwd labelling of the NCCL list relies on a CPU "backward" annotation that the trainer does not emit; read the
per-call durations in program order instead: forward gathers / reduce-scatters first, then the backward ones.)"""
import json
import sys
from collections import defaultdict


def analyze(path):
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")]
    gpu.sort(key=lambda e: e["ts"])
    t0, t1 = gpu[0]["ts"], max(e["ts"] + e["dur"] for e in gpu)
    # phases from the CPU annotations if present (ProfilerStep / backward)
    bwd = [e for e in cpu if e["name"].startswith("autograd::engine") or e["name"] == "torch.autograd.backward" or "backward" in e["name"].lower() and e["dur"] > 5e4]
    bwd_span = (min(e["ts"] for e in bwd), max(e["ts"] + e["dur"] for e in bwd)) if bwd else None
    busy = sum(e["dur"] for e in gpu)
    # union of GPU busy intervals (kernels on several streams may overlap)
    merged, cur = [], None
    for e in gpu:
        s, t = e["ts"], e["ts"] + e["dur"]
        if cur is None or s > cur[1]:
            if cur: merged.append(cur)
            cur = [s, t]
        else:
            cur[1] = max(cur[1], t)
    merged.append(cur)
    union = sum(t - s for s, t in merged)
    print(f"== {path}")
    print(f"   wall (first kernel -> last kernel end) {(t1 - t0)/1e3:8.1f} ms; GPU busy (union) {union/1e3:8.1f} ms = {100*union/(t1-t0):.0f} %; "
          f"summed kernel time {busy/1e3:.1f} ms")
    gaps = sorted(((merged[i + 1][0] - merged[i][1], merged[i][1]) for i in range(len(merged) - 1)), reverse=True)[:8]
    print("   largest GPU idle gaps (ms @ t ms): " + ", ".join(f"{g/1e3:.1f}@{(t-t0)/1e3:.0f}" for g, t in gaps))
    if bwd_span:
        bs, be = bwd_span
        in_b = [e for e in gpu if e["ts"] >= bs and e["ts"] < be]
        u_b = 0.0; cur = None
        for e in in_b:
            s, t = e["ts"], e["ts"] + e["dur"]
            if cur is None or s > cur[1]:
                if cur: u_b += cur[1] - cur[0]
                cur = [s, t]
            else:
                cur[1] = max(cur[1], t)
        if cur: u_b += cur[1] - cur[0]
        print(f"   backward (CPU span) {(be-bs)/1e3:.1f} ms; GPU busy inside it {u_b/1e3:.1f} ms ({100*u_b/(be-bs):.0f} %)")
    nccl = [e for e in gpu if "nccl" in e["name"].lower()]
    by = defaultdict(list)
    for e in nccl:
        key = e["name"].split("(")[0].replace("ncclDevKernel_", "").replace("ncclKernel_", "")
        phase = "bwd" if bwd_span and bs <= e["ts"] < be else "fwd"
        by[(phase, key)].append(e["dur"] / 1e3)
    for (phase, key), ds in sorted(by.items()):
        ds_s = " ".join(f"{d:.1f}" for d in ds[:24])
        print(f"   NCCL {phase} {key:28s} n={len(ds):3d} total {sum(ds):7.1f} ms: {ds_s}")
    cats = defaultdict(float)
    for e in gpu:
        n = e["name"]
        k = ("nccl" if "nccl" in n.lower() else "attn" if "_attn_" in n else "gemm" if ("gemm" in n or "cutlass" in n) else
             "spmm" if ("csrmm" in n or "cusparse" in n) else "memcpy" if e["cat"] != "kernel" else "other")
        cats[k] += e["dur"] / 1e3
    print("   GPU time by category (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
