"""Load a gt3d 2D partition (see gt3d.data.partition) for one rank of a Grid2D.

The grid must use the partition's padded node count: Grid2D(p_rows, p_cols, num_nodes, n_pad=meta['n_pad']),
so that every grid chunk is a whole number of partition chunks.
"""
import json
import os

import torch

from gt3d.dist.grid2d import Grid2D


def read_meta(part_dir):
    return json.load(open(os.path.join(part_dir, "meta.json")))


def _chunks_of_range(lo, hi, chunk):
    return range(lo // chunk, -(-hi // chunk))


def load_partition(part_dir, grid: Grid2D, device, load_weights=True):
    """Returns dict with x, y, masks (owned chunk, zero-padded), and the local blocks:
    gcn = (edge_index_local [2,E], weight [E]) with dst in row-local, src in col-local indices,
    edges = (edge_index_local, None) for the attention (same edges; self loops kept)."""
    meta = read_meta(part_dir)
    pc = meta["chunk"]
    n = meta["num_nodes"]
    assert grid.n_pad == meta["n_pad"], (grid.n_pad, meta["n_pad"])
    # ---- owned chunk features / labels / masks (a grid chunk may span several partition chunks or a sub-range of one) ----
    lo, hi = grid.owned_range()
    xs, ys, ms = [], [], {"train": [], "valid": [], "test": []}
    for i in _chunks_of_range(lo, hi, pc):
        a, b = max(lo, i * pc) - i * pc, min(hi, (i + 1) * pc, n) - i * pc          # sub-range inside partition chunk i
        if b <= a:
            continue
        xs.append(torch.load(os.path.join(part_dir, "x", f"{i}.pt"), weights_only=False)[a:b])
        ys.append(torch.load(os.path.join(part_dir, "y", f"{i}.pt"), weights_only=False)[a:b])
        for name, sub in [("train", "mask_train"), ("valid", "mask_val"), ("test", "mask_test")]:
            ms[name].append(torch.load(os.path.join(part_dir, sub, f"{i}.pt"), weights_only=False)[a:b])
    def pad_rows(t, fill=0):
        out = t.new_full((grid.chunk,) + tuple(t.shape[1:]), fill)
        out[: t.shape[0]] = t
        return out
    f = meta["num_features"]
    x = pad_rows(torch.cat(xs) if xs else torch.zeros(0, f)).to(device)
    y = pad_rows(torch.cat(ys) if ys else torch.zeros(0, dtype=torch.long)).to(device)
    masks = {k: pad_rows(torch.cat(v) if v else torch.zeros(0, dtype=torch.bool), False).to(device) & grid.valid_mask
             for k, v in ms.items()}
    # ---- local block: dst in row block r (P_o owned chunks), src in column block c ----
    c_lo, c_hi = grid.col_block_range()
    src_chunks = sorted(set(_chunks_of_range(c_lo, c_hi, pc)))
    dst_chunks = set()
    for rlo, rhi in grid.row_block_ranges():
        dst_chunks |= set(_chunks_of_range(rlo, rhi, pc))          # a set: several owned ranges may share a partition chunk
    dst_chunks = sorted(dst_chunks)
    eis, ws = [], []
    for i in src_chunks:
        for j in dst_chunks:
            ei, w = torch.load(os.path.join(part_dir, "blocks", f"{i}_{j}.pt"), weights_only=False)
            eis.append(ei)
            if load_weights and w is not None:
                ws.append(w)
    ei = torch.cat(eis, dim=1)
    w = torch.cat(ws) if ws else None
    # keep exactly the edges of my block (partition chunks may be larger than grid chunks)
    r_dst = grid.owner_of(ei[1])[0]
    c_src = grid.owner_of(ei[0])[1]
    keep = (r_dst == grid.r) & (c_src == grid.c)
    ei = ei[:, keep]
    w = w[keep] if w is not None else None
    ei_local = torch.stack([grid.global_to_col_local(ei[0]), grid.global_to_row_local(ei[1])]).to(device)
    return dict(meta=meta, x=x, y=y, masks=masks, edge_index=ei_local,
                weight=w.to(device) if w is not None else None, num_local_edges=int(ei.shape[1]))
