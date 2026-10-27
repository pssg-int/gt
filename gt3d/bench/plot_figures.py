"""Paper figures from the measurements in RESULTS.md (2026-09-26 final tables) and the training logs.

    python bench/plot_figures.py            -> figures/*.pdf + *.png

Fig 1  strong scaling products / Reddit, best 2D grid vs 1D (P x 1), per hidden size
Fig 2  accuracy vs training time (products hidden 128/256/512, Reddit hidden 128/256)
Fig 3  TorchGT same-batch comparison (products 870K batch): per-epoch time vs GPUs
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
LOGS = os.path.join(os.path.dirname(OUT), "logs")

# ms/epoch, steady state; (4, 8, 16 GPUs)
SCALING = {   # LayerNorm (headline); BatchNorm numbers in SCALING_BN
    "products h80":  {"2D": [788, 492, 368], "1D": [818, 633, 523]},
    "products h128": {"2D": [1380, 925, 751], "1D": [1363, 1129, 1094]},   # checkpointed
    "products h256": {"2D": [3345, 2118, 1555], "1D": [3515, 2835, 2550]}, # checkpointed
    "Reddit h128":   {"2D": [300, 170, 90.8], "1D": [317, 208, 129]},
    "Reddit h256":   {"2D": [849, 467, 268], "1D": [877, 523, 363]},
}
SCALING_BN = {
    "products h80":  {"2D": [850, 563, 471], "1D": [882, 702, 644]},
    "products h128": {"2D": [1518, 788, 647], "1D": [1505, 933, 914]},
    "products h256": {"2D": [3629, 2325, 1889], "1D": [3814, 3032, 2672]},
    "Reddit h128":   {"2D": [313, 206, 136], "1D": [331, 223, 191]},
    "Reddit h256":   {"2D": [875, 500, 346], "1D": [903, 573, 433]},
}
GPUS = [4, 8, 16]
TORCHGT = {"gt3d (fused, bf16)": ([1, 4, 8, 16], [0.55, 0.22, 0.197, 0.160]),
           "TorchGT released (random blocks)": ([4], [0.83]),
           "TorchGT real topology": ([4], [8.7])}


def fill_h128():
    """Read the checkpointed 4-GPU hidden-128 cells from the rerun log if present."""
    import glob
    for f in sorted(glob.glob(os.path.join(LOGS, "sbatch_products_h128_4gpu_*.log"))):
        for line in open(f):
            m = re.search(r"STEADY grid=(\d+)x(\d+) hidden=128 .*ms_per_epoch=([\d.]+)", line)
            if m:
                key = "2D" if m.group(1) == "2" else "1D"
                SCALING["products h128"][key][0] = float(m.group(3))


def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ds in zip(axes, ("products", "Reddit")):
        for name, series in SCALING.items():
            if not name.startswith(ds):
                continue
            h = name.split()[1]
            for layout, ls, mk in (("2D", "-", "o"), ("1D", "--", "s")):
                ys = series[layout]
                pts = [(g, y / 1000) for g, y in zip(GPUS, ys) if y is not None]
                if pts:
                    ax.plot(*zip(*pts), ls, marker=mk, label=f"{h} {layout}")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks(GPUS); ax.set_xticklabels([str(g) for g in GPUS])
        ax.set_xlabel("GPUs (A100)"); ax.set_ylabel("s / epoch"); ax.set_title(f"ogbn-{ds}" if ds == "products" else ds)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_scaling.{ext}"), dpi=150)


PAT = re.compile(r"^epoch\s+(\d+)\s+loss\s+([\d.]+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)\s+test\s+([\d.]+)\s+\|\s+best val\s+([\d.]+)\s+test\s+([\d.]+)\s+@\s+(\d+)\s+\|\s+([\d.]+)\s+ms/epoch")


def runs_in_log(path):
    """Split a log with several appended runs into segments; each segment ends with its FINAL line (hidden size)."""
    segs, cur = [], []
    for line in open(path):
        m = PAT.match(line)
        if m:
            ep, ms, bt = int(m.group(1)), float(m.group(9)), float(m.group(7))
            cur.append((ep * ms / 1000 / 60, 100 * bt))
        elif line.startswith("FINAL"):
            h = re.search(r"hidden=(\d+)", line)
            segs.append((int(h.group(1)) if h else None, cur))
            cur = []
    if cur:
        segs.append((None, cur))
    return segs


def curve(path, hidden):
    for h, pts in runs_in_log(path):
        if h == hidden and pts:
            return zip(*pts)
    return [], []


def fig_accuracy():
    runs = {"products h128 (4 GPUs)": ("products_accuracy.log", 128), "products h256 (4 GPUs)": ("products_accuracy.log", 256),
            "products h512 (4 GPUs)": ("products_accuracy_h512.log", 512),
            "Reddit h128 (4 GPUs)": ("reddit_acc_h128.log", 128), "Reddit h256 (4 GPUs)": ("reddit_acc_h256.log", 256)}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, (f, hidden) in runs.items():
        p = os.path.join(LOGS, f)
        if not os.path.exists(p):
            continue
        xs, ys = curve(p, hidden)
        if not xs:
            continue
        ax = axes[0] if name.startswith("products") else axes[1]
        ax.plot(xs, ys, label=name)
    for ax, ref in zip(axes, ({"GCN 75.6": 75.64, "SAGE 78.3": 78.29}, {"GCN 95.0": 95.0, "SAINT-GAT 96.6": 96.6})):
        for k, v in ref.items():
            ax.axhline(v, color="gray", ls=":", lw=0.8); ax.text(ax.get_xlim()[1] * 0.98, v + 0.1, k, ha="right", fontsize=7, color="gray")
        ax.set_xlabel("training time (min)"); ax.set_ylabel("test acc. at best val (%)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_title("ogbn-products, full graph"); axes[1].set_title("Reddit, full graph")
    axes[0].set_ylim(70, 80); axes[1].set_ylim(90, 98)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_accuracy_time.{ext}"), dpi=150)


def fig_torchgt():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, (xs, ys) in TORCHGT.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 16]); ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlabel("GPUs"); ax.set_ylabel("s per TorchGT epoch (2 iterations)"); ax.set_title("products 870K-node batch, same Graphormer")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_torchgt_batch.{ext}"), dpi=150)


if __name__ == "__main__":
    fig_scaling(); fig_accuracy(); fig_torchgt()
    print("figures written to", OUT, os.listdir(OUT))
