"""Single-GPU full-graph Exphormer training on ogbn-arxiv, reproducing the official
GraphGPS recipe (configs/Exphormer/arxiv.yaml) with gt3d's attention backends.

Recipe: reverse edges + self loops, expander degree 3 (type 1 edges), GCN+Exphormer x4,
hidden 80, 2 heads, dropout 0.3, BatchNorm, AdamW lr 1e-3 wd 1e-3, cosine schedule with
5 warm-up epochs over 600 epochs, grad-norm clip 1.0, cross-entropy on the train split,
evaluation every epoch, report test accuracy at the best validation epoch.

  python -m gt3d.train.train_single_gpu --backend triton --epochs 600
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gt3d.models.exphormer import Exphormer, AttentionGraph  # noqa: E402
from gt3d.data.expander import random_regular_expander_edges  # noqa: E402

EXPHORMER_DIR = os.environ.get("EXPHORMER_DIR", "/pscratch/sd/c/cunyang/gnn/baseline/Exphormer")


def load_arxiv(device, add_self_loops=True):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(EXPHORMER_DIR, "dataset"))
    data = ds[0]
    split = ds.get_idx_split()
    n = data.num_nodes
    ei = data.edge_index
    ei = torch.cat([ei, ei.flip(0)], dim=1)                   # add_reverse_edges
    if add_self_loops:
        loops = torch.arange(n).view(1, -1).repeat(2, 1)
        ei = torch.cat([ei, loops], dim=1)                    # add_self_loops (arxiv.yaml)
    x = data.x.to(device)
    y = data.y.view(-1).to(device)
    return n, x, y, ei.to(device), {k: v.to(device) for k, v in split.items()}, int(y.max().item()) + 1


def cosine_with_warmup(step, warmup, total):
    """GraphGPS / HuggingFace cosine schedule with linear warm-up (per-epoch steps)."""
    if step < warmup:
        return max(1e-6, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, x, ei, graph, y, split):
    model.eval()
    logits = model(x, ei, graph)
    pred = logits.argmax(-1)
    return {k: (pred[idx] == y[idx]).float().mean().item() for k, idx in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="triton", choices=["ref", "torch", "triton"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--no_self_loops", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    n, x, y, ei, split, num_classes = load_arxiv(dev, add_self_loops=not args.no_self_loops)
    exp = random_regular_expander_edges(n, args.exp_deg, seed=args.seed, device=dev)
    graph = AttentionGraph(ei, exp, n)
    print(f"arxiv: N={n} E_gcn={ei.shape[1]} E_attn={graph.edge_index.shape[1]} classes={num_classes} "
          f"backend={args.backend}", flush=True)

    model = Exphormer(x.shape[1], args.hidden, num_classes, args.layers, args.heads,
                      dropout=args.dropout, gnn_dropout=args.dropout, backend=args.backend).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, args.warmup, args.epochs))

    best = {"val": -1.0, "test": 0.0, "epoch": -1}
    t_train, t_eval = 0.0, 0.0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, ei, graph)
        loss = F.cross_entropy(logits[split["train"]], y[split["train"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        torch.cuda.synchronize(); t1 = time.perf_counter(); t_train += t1 - t0
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            acc = evaluate(model, x, ei, graph, y, split)
            torch.cuda.synchronize(); t_eval += time.perf_counter() - t1
            if acc["valid"] > best["val"]:
                best = {"val": acc["valid"], "test": acc["test"], "epoch": epoch}
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch:4d} loss {loss.item():.4f} train {acc['train']:.4f} val {acc['valid']:.4f} "
                      f"test {acc['test']:.4f} | best val {best['val']:.4f} test {best['test']:.4f} @ {best['epoch']} "
                      f"| {t_train/(epoch+1)*1e3:.1f} ms/epoch train, lr {sched.get_last_lr()[0]:.2e}", flush=True)
    print(f"FINAL backend={args.backend} best_val={best['val']:.4f} test@best_val={best['test']:.4f} "
          f"epoch={best['epoch']} train_time={t_train:.1f}s ({t_train/args.epochs*1e3:.1f} ms/epoch) "
          f"eval_time={t_eval:.1f}s peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
