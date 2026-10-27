"""Communication sparsity of the gather layouts: which rows of a gathered block a rank actually needs (CPU, minutes).

Rank (r, c) of a P_d x P_o block grid (the fused 2D grid; the (P_d, P_o) block grid of a head grid (P_d, P_o, P_i), whose
chunks are P_i times smaller but whose blocks are the same; GP-AG = (P, 1)) holds the tile A[r, c]: N/P_d destination
rows (row block) x N/P_o source columns (column block).  A row of x_col (K/V side) that no edge of the tile uses need not
be received, and a row of x_row (Q side) with no edge in the tile need not be received and its (acc, l) partial is zero.
So the sparsity a sparse gather could exploit is exactly "how many columns / rows of my share of A have no non-zero":

  column sparsity = share of the column block's sources with no edge in the tile  (x_col gather, dx_col reduce-scatter)
  row sparsity    = share of the row block's destinations with no edge in the tile (x_row gather, (acc, l) reduce-scatter)

Both are reported on the non-owned part of the block (the own chunk is local anyway): that share is what a sparse
collective would save of the bytes of that collective.  Counted on the attention edge set the trainer builds
(graph edges + self loops + expander edges of degree exp_deg), in the native id order, after a random permutation of the
ids (a partition without locality), and after an RCM reordering of the graph (a locality-preserving order), the latter two
on a subset of the grids.  The ring (P x 1): a chunk is forwarded to every rank, so a hop must carry the union of the rows
some other rank needs; that share is reported too.  Also the per-tile edge balance (max / mean), which a locality order
trades against.

    python bench/comm_sparsity.py --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --name products
    python bench/comm_sparsity.py --arxiv --name arxiv
Writes bench/comm_sparsity_<name>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.data.loader import read_meta  # noqa: E402
from gt3d.data.expander import feistel_expander_edges, random_regular_expander_edges  # noqa: E402

GRIDS = [(4, 1), (8, 1), (16, 1), (32, 1), (64, 1), (2, 2), (2, 4), (4, 4), (8, 4), (16, 4), (4, 2), (8, 2), (8, 8), (2, 8), (4, 16)]
SUBSET = [(16, 1), (64, 1), (4, 4), (16, 4), (8, 8)]
RING_P = [4, 16, 64]


def load_partition_edges(part_dir):
    meta = read_meta(part_dir)
    p = meta["p_part"]
    eis = []
    for i in range(p):
        for j in range(p):
            ei, _ = torch.load(os.path.join(part_dir, "blocks", f"{i}_{j}.pt"), weights_only=False)
            eis.append(ei)
    ei = torch.cat(eis, dim=1)
    return meta["num_nodes"], ei[0].numpy().astype(np.int64), ei[1].numpy().astype(np.int64)


def load_arxiv_edges():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt3d", "train"))
    from train_single_gpu import load_arxiv
    import functools
    _load = torch.load
    torch.load = functools.partial(_load, weights_only=False)      # the processed ogb data predates torch's safe loader
    try:
        n, _, _, ei, _, _ = load_arxiv(torch.device("cpu"))
    finally:
        torch.load = _load
    return n, ei[0].numpy().astype(np.int64), ei[1].numpy().astype(np.int64)


def pair_flags(a, b, nb):
    """Boolean [N, nb] table: flags[a, b] = True when the pair (a, b) occurs (a node id, b a block index)."""
    f = np.zeros(int(NMAX[0]) * nb, dtype=bool)
    f[a * nb + b] = True
    return f.reshape(-1, nb)


NMAX = [0]


def analyse(n, src, dst, pd, po):
    """Column / row emptiness of the P_d x P_o block grid on edges (src, dst) with global ids."""
    world = pd * po
    chunk = -(-n // world)
    ids = np.arange(n, dtype=np.int64)
    rb = lambda v: (v // chunk) % pd                      # noqa: E731  row block of a node (its owner's r)
    cb = lambda v: (v // chunk) // pd                     # noqa: E731  column block of a node (its owner's c)
    out = {}
    # ---- columns: does source s have an edge into row block r?  flags[s, r] ----
    f = pair_flags(src, rb(dst), pd)[:n]
    u_all = int(f.sum())
    u_own = int(f[ids, rb(ids)].sum())
    out["col_nonempty_all"] = u_all / (n * pd)
    out["col_empty_nonowned"] = 1.0 - (u_all - u_own) / (n * (pd - 1)) if pd > 1 else 0.0
    # per tile (r, c): sources of column block c (a contiguous id range) with an edge into row block r
    per = np.zeros(world)
    for c in range(po):
        lo, hi = c * pd * chunk, min((c + 1) * pd * chunk, n)
        per[np.arange(pd) * po + c] = f[lo:hi].sum(0) / (pd * chunk)
    out["col_empty_tile_min"], out["col_empty_tile_max"] = float(1 - per.max()), float(1 - per.min())
    # ---- rows: does destination d have an edge from column block c?  flags[d, c] ----
    f = pair_flags(dst, cb(src), po)[:n]
    u_all = int(f.sum())
    u_own = int(f[ids, cb(ids)].sum())
    out["row_nonempty_all"] = u_all / (n * po)
    out["row_empty_nonowned"] = 1.0 - (u_all - u_own) / (n * (po - 1)) if po > 1 else 0.0
    per = np.zeros(world)
    rbi = rb(ids)
    for r in range(pd):
        sel = rbi == r
        per[r * po + np.arange(po)] = f[sel].sum(0) / (po * chunk)
    out["row_empty_tile_min"], out["row_empty_tile_max"] = float(1 - per.max()), float(1 - per.min())
    # ---- edge balance over the tiles ----
    e = np.bincount(rb(dst) * po + cb(src), minlength=world)
    out["edges_tile_max_over_mean"] = float(e.max() / e.mean())
    out["edges_tile_min_over_mean"] = float(e.min() / e.mean())
    return out


def ring_share(n, src, dst, p):
    """Share of a chunk's rows that at least one other rank needs (what a ring hop must carry), P x 1."""
    chunk = -(-n // p)
    other = (dst // chunk) != (src // chunk)
    f = np.zeros(n, dtype=bool)
    f[src[other]] = True
    return float(f.mean())


def rcm_order(n, src, dst):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    a = sp.csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    a = a + a.T
    perm = reverse_cuthill_mckee(a, symmetric_mode=True)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    return inv                                                   # inv[old id] = new id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--arxiv", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_rcm", action="store_true")
    args = ap.parse_args()
    say = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.perf_counter()
    if args.arxiv:
        n, gs, gd = load_arxiv_edges()
        exp = random_regular_expander_edges(n, args.exp_deg, seed=args.seed)
    else:
        n, gs, gd = load_partition_edges(args.data)
        exp = feistel_expander_edges(n, args.exp_deg, seed=args.seed)
    es, ed = exp[0].numpy().astype(np.int64), exp[1].numpy().astype(np.int64)
    NMAX[0] = -(-n // 1024) * 1024
    say(f"[sparsity] {args.name}: N={n} graph edges (with self loops) {len(gs)} expander edges {len(es)} loaded in {time.perf_counter() - t0:.0f} s")
    deg = np.bincount(np.concatenate([gd, ed]), minlength=n)
    gdeg = np.bincount(gd, minlength=n)
    out = dict(name=args.name, num_nodes=n, graph_edges=int(len(gs)), expander_edges=int(len(es)), exp_deg=args.exp_deg,
               in_degree=dict(attn_mean=float(deg.mean()), attn_min=int(deg.min()), graph_mean=float(gdeg.mean()),
                              attn_le={str(k): float((deg <= k).mean()) for k in [7, 8, 10, 12, 16, 24, 32, 64]},
                              graph_le={str(k): float((gdeg <= k).mean()) for k in [1, 2, 3, 4, 6, 8, 16]}),
               variants={})
    variants = [("native", gs, gd, es, ed, GRIDS), ("native_graph_only", gs, gd, None, None, SUBSET)]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n).astype(np.int64)
    variants.append(("random", perm[gs], perm[gd], perm[es], perm[ed], SUBSET))
    if not args.no_rcm:
        t1 = time.perf_counter()
        try:
            inv = rcm_order(n, gs, gd)
            say(f"[sparsity] RCM order in {time.perf_counter() - t1:.0f} s")
            variants.append(("rcm", inv[gs], inv[gd], inv[es], inv[ed], SUBSET))
            variants.append(("rcm_graph_only", inv[gs], inv[gd], None, None, SUBSET))
        except Exception as e:  # noqa: BLE001
            say(f"[sparsity] RCM failed: {e!r}")
    for name, s1, d1, s2, d2, grids in variants:
        src = s1 if s2 is None else np.concatenate([s1, s2])
        dst = d1 if s2 is None else np.concatenate([d1, d2])
        res = {}
        for pd, po in grids:
            t1 = time.perf_counter()
            r = analyse(n, src, dst, pd, po)
            res[f"{pd}x{po}"] = r
            say(f"[sparsity] {args.name} {name:17s} {pd:3d}x{po:<3d} col empty (non-owned) {r['col_empty_nonowned']:.3f} "
                f"[tiles {r['col_empty_tile_min']:.3f}-{r['col_empty_tile_max']:.3f}]  row empty (non-owned) {r['row_empty_nonowned']:.3f} "
                f"[{r['row_empty_tile_min']:.3f}-{r['row_empty_tile_max']:.3f}]  edges/tile max/mean {r['edges_tile_max_over_mean']:.2f} "
                f"min/mean {r['edges_tile_min_over_mean']:.2f}  ({time.perf_counter() - t1:.0f} s)")
        if name in ("native", "random", "rcm"):
            res["ring"] = {str(p): ring_share(n, src, dst, p) for p in RING_P}
            say(f"[sparsity] {args.name} {name:17s} ring: share of a chunk needed by some other rank " +
                " ".join(f"P={p}: {v:.3f}" for p, v in res["ring"].items()))
        out["variants"][name] = res
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"comm_sparsity_{args.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    say(f"[sparsity] written {path} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
