"""Read a Plexus partition directory (plexus.utils.dataset.partition_graph_2d output) for one rank of a Grid2D.

Plexus layout (p x p node chunks of size pc = pad(N, p) / p, feature chunks of near-equal size):
    metadata.pt                    (num_nodes, num_features, num_classes)
    edge_index/0/{i}_{j}.pt        (edge_index [2, E] global ids, edge_weight [E]) with edge_index[0] in node chunk i
                                   and edge_index[1] in node chunk j  (GCN-normalised weights, self loops included)
    input_features/{i}_{j}.pt      node chunk i x feature chunk j
    output_labels/0/{i}.pt, masks/{train,val,test}/0/{i}.pt
Double-permutation datasets (edge_index/1 present) are refused: they carry two node orderings.

The returned dict has the same contract as gt3d.data.loader.load_partition (owned x/y/masks, zero-padded to the grid
chunk; the local block as (col-local src, row-local dst) edges with weights), so the trainer can use either format.
Nothing in Plexus is modified or imported for this; only its on-disk format is read.
"""
import os

import torch

from gt3d.dist.grid2d import Grid2D


def read_plexus_meta(part_dir):
    num_nodes, num_features, num_classes = torch.load(os.path.join(part_dir, "metadata.pt"), weights_only=False)
    p = len([f for f in os.listdir(os.path.join(part_dir, "output_labels", "0")) if f.endswith(".pt")])
    if os.path.isdir(os.path.join(part_dir, "edge_index", "1")):
        raise ValueError(f"{part_dir} is a double-permutation Plexus partition (edge_index/1): not usable by gt3d")
    pc = -(-int(num_nodes) // p)                                     # pad_dimension(N, p) // p
    n_pad_plexus = pc * p
    return dict(num_nodes=int(num_nodes), num_features=int(num_features), num_classes=int(num_classes),
                p_part=p, chunk=pc, n_pad=n_pad_plexus, has_weight=True)


def grid_n_pad(meta, world, multiple=1):
    """Padded node count for a Grid2D over `world` ranks reading this partition: a multiple of world (and of
    `multiple`, e.g. 4096 to keep the block layout grid-agnostic like gt3d's own partitions)."""
    import math
    m = math.lcm(world, multiple)
    return -(-meta["num_nodes"] // m) * m


def _chunks_of_range(lo, hi, chunk):
    return range(lo // chunk, -(-hi // chunk))


def load_plexus_partition(part_dir, grid: Grid2D, device, load_weights=True, src_first=False):
    """src_first=False: edge_index[0] is the destination (row of A) as in Plexus's DataLoader.__split_adj
    (A[dst, src], AGG = A @ H); src_first=True: PyG order (edge_index[0] = source)."""
    meta = read_plexus_meta(part_dir)
    pc, p, n = meta["chunk"], meta["p_part"], meta["num_nodes"]
    assert grid.n_pad >= n, (grid.n_pad, n)
    f = meta["num_features"]
    f_sizes = [f // p + (1 if k < f % p else 0) for k in range(p)]     # tensor_split-like feature chunks
    # ---- owned rows: features (all feature chunks of every node chunk that intersects the owned range), labels, masks
    lo, hi = grid.owned_range()
    xs, ys, ms = [], [], {"train": [], "valid": [], "test": []}
    for i in _chunks_of_range(lo, min(hi, n), pc):
        if i >= p:
            break
        a, b = max(lo, i * pc) - i * pc, min(hi, (i + 1) * pc, n) - i * pc
        if b <= a:
            continue
        parts = [torch.load(os.path.join(part_dir, "input_features", f"{i}_{j}.pt"), weights_only=False)[a:b] for j in range(p)]
        xs.append(torch.cat(parts, dim=1))
        ys.append(torch.load(os.path.join(part_dir, "output_labels", "0", f"{i}.pt"), weights_only=False)[a:b].view(-1))
        for name, sub in [("train", "train"), ("valid", "val"), ("test", "test")]:
            ms[name].append(torch.load(os.path.join(part_dir, "masks", sub, "0", f"{i}.pt"), weights_only=False)[a:b])

    def pad_rows(t, fill=0):
        out = t.new_full((grid.chunk,) + tuple(t.shape[1:]), fill)
        out[: t.shape[0]] = t
        return out

    x = pad_rows(torch.cat(xs) if xs else torch.zeros(0, f)).to(device)
    y = pad_rows(torch.cat(ys) if ys else torch.zeros(0, dtype=torch.long)).to(device)
    masks = {k: pad_rows(torch.cat(v) if v else torch.zeros(0, dtype=torch.bool), False).to(device) & grid.valid_mask
             for k, v in ms.items()}
    # ---- local block: dst in my row block, src in my column block
    c_lo, c_hi = grid.col_block_range()
    src_chunks = sorted({i for i in _chunks_of_range(c_lo, min(c_hi, n), pc) if i < p})
    dst_chunks = set()
    for rlo, rhi in grid.row_block_ranges():                   # Grid2D: P_o owned ranges; Grid3D: P_o * P_i
        dst_chunks |= {i for i in _chunks_of_range(rlo, min(rhi, n), pc) if i < p}   # a set: owned ranges may share a chunk
    dst_chunks = sorted(dst_chunks)
    eis, ws = [], []
    for i in src_chunks:
        for j in dst_chunks:
            name = f"{i}_{j}.pt" if src_first else f"{j}_{i}.pt"
            ei, w = torch.load(os.path.join(part_dir, "edge_index", "0", name), weights_only=False)
            if not src_first:
                ei = ei.flip(0)                                        # -> (src, dst)
            eis.append(ei)
            if load_weights and w is not None:
                ws.append(w)
    ei = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    w = torch.cat(ws) if ws else None
    r_dst = grid.owner_of(ei[1])[0]                           # (r, c[, h]) on a Grid2D / Grid3D
    c_src = grid.owner_of(ei[0])[1]
    keep = (r_dst == grid.r) & (c_src == grid.c)
    ei = ei[:, keep]
    w = w[keep] if w is not None else None
    ei_local = torch.stack([grid.global_to_col_local(ei[0]), grid.global_to_row_local(ei[1])]).to(device)
    meta_out = dict(meta, num_edges=None, self_loops=None)
    return dict(meta=meta_out, x=x, y=y, masks=masks, edge_index=ei_local,
                weight=w.to(device) if w is not None else None, num_local_edges=int(ei.shape[1]))
