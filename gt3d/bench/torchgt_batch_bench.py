"""Same model, same batch as TorchGT: time one training iteration (forward + backward + optimizer step) of
TorchGT's node-level Graphormer (GPH_slim: 4 layers, hidden 64, ffn 64, 8 heads) on the TorchGT batch
construction (seq_len random train nodes, induced 1-hop edges, global token 0 connected to all) with gt3d.

  single GPU:  python bench/torchgt_batch_bench.py --dataset arxiv --seq_len 64000
  4 GPUs    :  torchrun --nproc_per_node 4 bench/torchgt_batch_bench.py --dataset products --seq_len 870000 --grid 2 2

TorchGT reference numbers (torchgt_profiling_report.md, 4xA100, GPU part of an epoch = 2 iterations):
arxiv 64K  released/random blocks 0.057 s, paper-faithful 0.55 s, real topology + global token 0.56 s;
products 870K released 0.83 s, real topology 8.7 s.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.models.graphormer_node import GraphormerNode, DistGraphormerNode  # noqa: E402

EXPHORMER_DIR = "/pscratch/sd/c/cunyang/gnn/baseline/Exphormer"
PRODUCTS_PT = "/pscratch/sd/c/cunyang/gnn/plexus/dataset/products/processed/processed_products.pt"


def load(dataset, split="ogb"):
    """split='ogb': official training split; split='random': TorchGT's seeded random 6/2/2 split of all nodes
    (TorchGT's products runs use the random split, which is why its 870K-node batches exist at all)."""
    x, y, ei, train_idx, ncls = _load_raw(dataset)
    if split == "random":
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(x.shape[0], generator=g)
        train_idx = perm[: int(0.6 * x.shape[0])]
    return x, y, ei, train_idx, ncls


def _load_raw(dataset):
    if dataset == "arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(EXPHORMER_DIR, "dataset"))
        d = ds[0]
        ei = torch.cat([d.edge_index, d.edge_index.flip(0)], dim=1)       # TorchGT makes the graph undirected
        return d.x, d.y.view(-1), ei, ds.get_idx_split()["train"], int(d.y.max()) + 1
    data, meta = torch.load(PRODUCTS_PT, weights_only=False)
    num_classes = int(meta[0]) if isinstance(meta, (tuple, list)) else int(meta)
    ei = data.edge_index[:, data.edge_index[0] != data.edge_index[1]]
    return data.x, data.y.view(-1), ei, data.train_mask.nonzero().view(-1), num_classes


def torchgt_batch(x, y, ei, train_idx, seq_len, seed=0):
    """First batch of TorchGT's epoch: shuffled train indices, induced subgraph, global token 0."""
    from torch_geometric.utils import subgraph
    g = torch.Generator().manual_seed(seed)
    idx = train_idx[torch.randperm(train_idx.numel(), generator=g)][:seq_len]
    sub_ei, _ = subgraph(idx, ei, num_nodes=x.shape[0], relabel_nodes=True)
    n = idx.numel()
    ar = torch.arange(n) + 1
    z = torch.zeros(n, dtype=torch.long)
    virt = torch.cat([torch.stack([ar, z]), torch.stack([z, ar])], dim=1)     # fix_edge_index
    full_ei = torch.cat([sub_ei + 1, virt], dim=1)
    return x[idx], y[idx], full_ei, n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv", choices=["arxiv", "products"])
    ap.add_argument("--seq_len", type=int, default=64000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--grid", type=int, nargs=2, default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--comm_bf16", action="store_true")
    ap.add_argument("--fused", action="store_true", help="fused-communication blocks (gather LN(x) once, one reduce-scatter)")
    ap.add_argument("--rs_bf16", action="store_true", help="(fused) reduce-scatter the partials in bfloat16")
    ap.add_argument("--split", default="ogb", choices=["ogb", "random"])
    args = ap.parse_args()
    dist_mode = args.grid is not None
    if dist_mode:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank(); torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank = 0
    dev = torch.device("cuda")
    x, y, ei, train_idx, num_classes = load(args.dataset, args.split)
    xb, yb, eb, n = torchgt_batch(x, y, ei, train_idx, args.seq_len)
    if rank == 0:
        print(f"{args.dataset}: batch nodes {n-1} (+1 global token) edges {eb.shape[1]} (incl. {2*(n-1)} global-token edges) "
              f"model {args.layers}L/h{args.hidden}/ffn{args.ffn}/{args.heads} heads", flush=True)
    torch.manual_seed(0)
    if not dist_mode:
        model = GraphormerNode(x.shape[1], args.hidden, args.ffn, args.heads, args.layers, num_classes, eb.to(dev), n).to(dev)
        xg, yg = xb.to(dev), yb.to(dev)
        def step():
            out = model(xg)
            loss = torch.nn.functional.cross_entropy(out, yg)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    else:
        from gt3d.dist.grid2d import Grid2D
        from gt3d.dist.layers import dist_cross_entropy, allreduce_grads
        grid = Grid2D(args.grid[0], args.grid[1], n).to(dev)
        blk, _ = grid.local_block(eb)
        model = DistGraphormerNode(grid, (blk.to(dev), None), x.shape[1], args.hidden, args.ffn, args.heads, args.layers,
                                   num_classes, comm_dtype=torch.bfloat16 if args.comm_bf16 else None,
                                   fused=args.fused, rs_dtype=torch.bfloat16 if args.rs_bf16 else None).to(dev)
        for p in model.parameters():
            dist.broadcast(p.data, 0)
        xfull = torch.cat([torch.zeros(1, x.shape[1]), xb]); yfull = torch.cat([torch.zeros(1, dtype=torch.long), yb])
        xg, yg = grid.take_owned(xfull).to(dev), grid.take_owned(yfull).to(dev)
        mask = grid.valid_mask.clone(); lo, _ = grid.owned_range()
        if lo == 0:
            mask[0] = False                                                     # the global token has no label
        params = list(model.parameters())
        def step():
            out = model(xg)
            loss = dist_cross_entropy(out, yg, mask, n - 1)
            loss.backward(); allreduce_grads(params); opt.step(); opt.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.iters
    if rank == 0:
        print(f"RESULT {args.dataset} split={args.split} seq_len={args.seq_len} {'grid=%dx%d' % tuple(args.grid) if dist_mode else '1 GPU'} "
              f"{'bf16comm ' if args.comm_bf16 else ''}{'fused ' if args.fused else ''}{'rs_bf16 ' if args.rs_bf16 else ''}iteration {dt*1e3:.1f} ms  (TorchGT epoch = 2 iterations -> {2*dt:.3f} s)  "
              f"peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
    if dist_mode:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
