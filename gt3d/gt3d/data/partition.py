"""Static 2D partitioning of a full graph into P x P blocks in ONE node-id space (gt3d format).

Input: a PyG-style Data object (or a Plexus `processed_*.pt` tuple) with x, y, edge_index
(src -> dst), optional edge_weight (GCN-normalised, self loops included), train/val/test masks.
Nodes are padded to N_pad = pad(N, p_part); chunk = N_pad / p_part.  Output directory:

    meta.json                      num_nodes, num_features, num_classes, p_part, chunk, n_pad, has_weight
    blocks/{i}_{j}.pt              (edge_index [2, E_ij] GLOBAL ids, weight [E_ij] or None) with src in chunk i, dst in chunk j
    x/{i}.pt  y/{i}.pt             feature / label rows of chunk i (unpadded length for the last chunk)
    mask_{train,val,test}/{i}.pt   boolean masks of chunk i

Any Grid2D whose chunk is a multiple of `chunk` (P divides p_part) can load it, see gt3d.data.loader.

    python -m gt3d.data.partition --src /pscratch/sd/c/cunyang/gnn/plexus/dataset/products/processed/processed_products.pt \
        --out /pscratch/sd/c/cunyang/gnn/gt/data/products_p16 --p_part 16
"""
import argparse
import gc
import json
import math
import os
import time

import torch


def pad_dimension(n, p):
    return -(-n // p) * p


def partition_2d(data, num_classes, out_dir, p_part, drop_self_loops_in_blocks=False, pad_multiple=4096):
    n, f = data.x.shape
    # pad so that n_pad is a multiple of p_part AND of pad_multiple: any grid with P | pad_multiple then has
    # integer chunks (grid chunks are unions or sub-ranges of partition chunks; see gt3d.data.loader)
    n_pad = pad_dimension(n, p_part * pad_multiple // math.gcd(p_part, pad_multiple))
    chunk = n_pad // p_part
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["blocks", "x", "y", "mask_train", "mask_val", "mask_test"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    # 1) per-node arrays first (then freed): x, y, masks  (skipped when already present from an earlier run)
    y_all = data.y.view(-1)
    masks_all = [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]
    for i in range(p_part):
        lo, hi = i * chunk, min((i + 1) * chunk, n)
        fx = os.path.join(out_dir, "x", f"{i}.pt")
        if not os.path.exists(fx) or os.path.getsize(fx) == 0:
            torch.save(data.x[lo:hi].clone(), fx)
        torch.save(y_all[lo:hi].clone(), os.path.join(out_dir, "y", f"{i}.pt"))
        for name, m in masks_all:
            torch.save(m[lo:hi].clone(), os.path.join(out_dir, f"mask_{name}", f"{i}.pt"))
    print("per-node arrays written", flush=True)
    ei = data.edge_index
    w = getattr(data, "edge_weight", None)
    del data, y_all, masks_all
    gc.collect()
    if drop_self_loops_in_blocks:
        keep = ei[0] != ei[1]
        ei = ei[:, keep]
        w = w[keep] if w is not None else None
    # 2) edges, streamed by source chunk: no global sort, peak memory ~ E*(16+4) bytes + one chunk-row subset
    t0 = time.time()
    src_chunk = (ei[0] // chunk).to(torch.int32)
    counts = torch.zeros(p_part * p_part, dtype=torch.long)
    self_loops = 0
    for i in range(p_part):
        marker = os.path.join(out_dir, "blocks", f"row_{i}.done")
        if os.path.exists(marker):                                   # resume: this block row was completed by an earlier run
            done = json.load(open(marker))
            counts[i * p_part:(i + 1) * p_part] = torch.tensor(done["counts"])
            self_loops += int(done["self_loops"])
            print(f"  block row {i+1}/{p_part} already written, skipped", flush=True)
            continue
        sel = (src_chunk == i).nonzero(as_tuple=True)[0]
        ei_i = ei[:, sel]
        w_i = w[sel] if w is not None else None
        del sel
        dst_chunk = (ei_i[1] // chunk)
        order = torch.argsort(dst_chunk)
        ei_i, dst_chunk = ei_i[:, order], dst_chunk[order]
        w_i = w_i[order] if w_i is not None else None
        del order
        cnt = torch.bincount(dst_chunk, minlength=p_part)
        off = torch.zeros(p_part + 1, dtype=torch.long)
        off[1:] = torch.cumsum(cnt, 0)
        row_loops = 0
        for j in range(p_part):
            a, b = off[j].item(), off[j + 1].item()
            blk = ei_i[:, a:b].clone()
            row_loops += int((blk[0] == blk[1]).sum())
            dst = os.path.join(out_dir, "blocks", f"{i}_{j}.pt")
            tmp = f"{dst}.tmp{os.getpid()}"
            torch.save((blk, w_i[a:b].clone() if w_i is not None else None), tmp)
            os.replace(tmp, dst)                                     # atomic: a file that exists is complete
        self_loops += row_loops
        counts[i * p_part:(i + 1) * p_part] = cnt
        json.dump(dict(counts=cnt.tolist(), self_loops=row_loops), open(marker + ".tmp", "w"))
        os.replace(marker + ".tmp", marker)
        del ei_i, w_i, dst_chunk
        gc.collect()
        print(f"  wrote block row {i+1}/{p_part} ({time.time()-t0:.0f}s)", flush=True)
    print(f"blocks written in {time.time()-t0:.1f}s; block edges min/max {counts.min().item()}/{counts.max().item()}", flush=True)
    num_edges = int(ei.shape[1])
    has_weight = w is not None
    meta = dict(num_nodes=int(n), num_features=int(f), num_classes=int(num_classes), p_part=int(p_part),
                chunk=int(chunk), n_pad=int(n_pad), num_edges=num_edges, has_weight=has_weight,
                self_loops=int(self_loops))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("meta:", meta, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Plexus processed_*.pt (tuple (Data, (num_classes,))) or a PyG Data .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_part", type=int, default=16)
    ap.add_argument("--pad_multiple", type=int, default=4096)
    args = ap.parse_args()
    obj = torch.load(args.src, weights_only=False)
    if isinstance(obj, (tuple, list)):
        data, meta = obj[0], obj[1]
        num_classes = int(meta[0]) if isinstance(meta, (tuple, list)) else int(meta)
    else:
        data, num_classes = obj, int(obj.y.max().item()) + 1
    print(data, "classes", num_classes, flush=True)
    partition_2d(data, num_classes, args.out, args.p_part, pad_multiple=args.pad_multiple)


if __name__ == "__main__":
    main()
