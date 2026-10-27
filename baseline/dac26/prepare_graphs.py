"""Extract the three graphs of the DAC'26 evaluation (ogbn-proteins, ogbn-products, Reddit) as plain tensors
{"num_nodes": int, "edge_index": int64 [2, E]} for the DGL re-implementation (one-time, CPU, plexus_env).

    source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
    python prepare_graphs.py --out graphs

Edge counts should match the paper's Table 2: proteins 132K / 79M, products 2.4M / 123M, Reddit 233K / 114M.
"""
import argparse
import os

import torch

PLEXUS = "/pscratch/sd/c/cunyang/gnn/plexus/dataset"
TORCHGT = "/pscratch/sd/c/cunyang/gnn/gt/baseline/torchgt/dataset"


def load(name):
    if name == "products":                                   # OGB numbering, both directions (TorchGT copy)
        ei = torch.load(os.path.join(TORCHGT, "ogbn-products", "edge_index.pt"))
        n = int(torch.load(os.path.join(TORCHGT, "ogbn-products", "y.pt")).shape[0])
        return n, ei
    if name == "reddit":
        from torch_geometric.datasets import Reddit
        data = Reddit(os.path.join(PLEXUS, "reddit", "raw"))[0]
        return int(data.num_nodes), data.edge_index
    if name == "proteins":
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset("ogbn-proteins", root=os.path.join(PLEXUS, "proteins", "raw"))[0]
        return int(data.num_nodes), data.edge_index
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="graphs")
    ap.add_argument("--graphs", nargs="+", default=["proteins", "reddit", "products"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.graphs:
        n, ei = load(name)
        ei = ei.to(torch.int64).contiguous()
        loops = int((ei[0] == ei[1]).sum())
        print(f"{name}: N={n} E={ei.shape[1]} self_loops={loops}", flush=True)
        torch.save({"num_nodes": n, "edge_index": ei}, os.path.join(args.out, f"{name}.pt"))


if __name__ == "__main__":
    main()
