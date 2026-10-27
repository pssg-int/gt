"""Tables from the 16-node NCCL diagnosis (scripts/sbatch_rs_diag_64.sh): per layout and per-rank chunk size, the
reduce-scatter / all-gather / all-to-all rate (GB/s of received bytes per rank) of every run tag, side by side.

    python bench/analyze_rs_diag.py [--kind rs] [--tags _64bb,_64iso,...]
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, f"hw_constants_nccl{tag}.json")
    return json.load(open(p))["coll"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="rs", choices=["rs", "ag", "a2a"])
    ap.add_argument("--tags", default=None, help="comma list of tags (default: every hw_constants_nccl_64*.json, plus the 4-node table '')")
    args = ap.parse_args()
    if args.tags:
        tags = args.tags.split(",")
    else:
        tags = [""] + sorted(os.path.basename(p)[len("hw_constants_nccl"):-5] for p in glob.glob(os.path.join(HERE, "hw_constants_nccl_64*.json")))
    data = {t: load(t) for t in tags}
    data = {t: d for t, d in data.items() if d}
    layouts = sorted({l for d in data.values() for l in d}, key=lambda s: (int(s.split("x")[1]), int(s.split("x")[0])))
    sizes = sorted({float(mb) for d in data.values() for l in d.values() for mb in l["by_mb"]})
    key = f"{args.kind}_gbs"
    print(f"{args.kind.upper()} GB/s per rank (received bytes / time); columns = run tags ('' = 4-node table of 10-17)")
    print(f"{'layout':7s} {'MB':>6s} " + " ".join(f"{(t or '4node'):>12s}" for t in data))
    for lay in layouts:
        for mb in sizes:
            row = []
            for t, d in data.items():
                r = d.get(lay, {}).get("by_mb", {}).get(str(mb)) or d.get(lay, {}).get("by_mb", {}).get(str(int(mb)) if mb == int(mb) else None)
                row.append(f"{r[key]:12.1f}" if r else f"{'':>12s}")
            if any(c.strip() for c in row):
                print(f"{lay:7s} {mb:6.2f} " + " ".join(row))
        print()


if __name__ == "__main__":
    main()
