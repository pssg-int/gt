"""Per-graph locality coefficients of the attention and SpMM kernels for the white-box model (one GPU, minutes).

The kernel constants of grid_select.py (bench/kernel_constants.py) come from uniform-random CSR blocks.  A real graph's
block has locality (partition order, degree skew), so its per-edge cost differs: the K / V rows a block row gathers are
re-used across neighbouring rows and hit in L2 more often.  This script rebuilds the rank-0 block of several 2D grids of
the graph from its partition files (the same block the trainer builds: GCN edges + self loops + the expander edges), times
the attention kernels (forward, backward) and the cuSPARSE SpMM (A x, A^T g) on it, and on a uniform-random block of
exactly the same rows / columns / edges / edge-type mix.  The ratio

    kappa = T(real block) / T(random block of the same shape)

is the graph's locality coefficient; grid_select multiplies the per-edge term of attn_ms / spmm_ms by it (interpolated in
the block's K / V footprint, cols x d x 4 B, the quantity that decides the L2 behaviour).  Nothing is fitted: one measured
number per graph, direction and block footprint.

    srun --jobid=<id> -n 1 --gpus-per-node 1 bash scripts/run_on_node.sh python bench/graph_constants.py \\
        --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --name products --configs 4x32,8x32,2x40 --grids 2x2,4x4,8x8,16x4,4x16
Writes bench/graph_constants_<name>.json; grid_select.py reads it (Graph(name=...)): the measured real-block time replaces the
constants x counts term for a block shape and head configuration that was measured, the ratio measured / model of the nearest
measured footprint scales the term for the others.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402
from gt3d.data.loader import load_partition, read_meta  # noqa: E402
from gt3d.data.expander import block_expander_edges  # noqa: E402


class BlockGeometry:
    """Rank (r, c) of a P_d x P_o Grid2D (row-major layout) without torch.distributed: the geometry methods the loader and
    the expander need (copied from gt3d.dist.grid2d.Grid2D)."""

    def __init__(self, p_rows, p_cols, r, c, num_nodes, n_pad):
        self.p_rows, self.p_cols, self.r, self.c = p_rows, p_cols, r, c
        self.num_nodes = num_nodes
        world = p_rows * p_cols
        assert n_pad % world == 0, (n_pad, world)
        self.chunk, self.n_pad = n_pad // world, n_pad
        lo, hi = self.owned_range()
        self.valid_mask = torch.arange(lo, hi) < num_nodes

    def to(self, device):
        self.valid_mask = self.valid_mask.to(device)
        return self

    def chunk_of(self, r, c):
        return c * self.p_rows + r

    def owned_range(self, r=None, c=None):
        r = self.r if r is None else r
        c = self.c if c is None else c
        k = self.chunk_of(r, c)
        return k * self.chunk, (k + 1) * self.chunk

    def owner_of(self, node_ids):
        k = node_ids // self.chunk
        return k % self.p_rows, k // self.p_rows

    def col_block_range(self, c=None):
        c = self.c if c is None else c
        return c * self.p_rows * self.chunk, (c + 1) * self.p_rows * self.chunk

    def row_block_ranges(self, r=None):
        r = self.r if r is None else r
        return [self.owned_range(r, c) for c in range(self.p_cols)]

    @property
    def n_row_block(self):
        return self.p_cols * self.chunk

    def global_to_row_local(self, gid):
        r, c = self.owner_of(gid)
        return c * self.chunk + (gid - self.chunk_of(r, c) * self.chunk)

    def global_to_col_local(self, gid):
        lo, _ = self.col_block_range(self.owner_of(gid)[1])
        return gid - lo


def timeit(fn, iters=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


RETUNE = True


def retune():
    """Forget the autotuned launch configurations of the attention kernels, so that the next call tunes for THIS block.
    The autotune key is (H, D, score, norm), not the block's density: a training process tunes on its own block, but one
    process timing every grid would otherwise run the sparse blocks with the configuration tuned on the first (densest)
    one (BLOCK_E 64: 3x slower rows below 16 edges, bench/autotune_order.py)."""
    from gt3d.attention import triton_tiled as tt
    for mod in (tk, tt):
        for name in dir(mod):
            fn = getattr(mod, name)
            if hasattr(fn, "configs") and isinstance(getattr(fn, "cache", None), dict):
                fn.cache.clear()


def time_block(ei, et, w, R, C, h, dh, dev, gen, label, say):
    """Attention fwd / bwd and SpMM A x / A^T g on the block (col-local src, row-local dst)."""
    if RETUNE:
        retune()
    E = int(ei.shape[1])
    crow, col, etype, perm = build_csr(ei, R, et)
    crow32, col32, et32 = crow.to(torch.int32).contiguous(), col.to(torch.int32), etype.to(torch.int32)
    d = h * dh
    q = torch.randn(R, h, dh, device=dev, generator=gen); k = torch.randn(C, h, dh, device=dev, generator=gen)
    v = torch.randn(C, h, dh, device=dev, generator=gen); gate = torch.randn(2, h, dh, device=dev, generator=gen)
    dacc = torch.randn(R, h, dh, device=dev, generator=gen); dl = torch.randn(R, h, device=dev, generator=gen)
    m0 = torch.zeros(R, h, device=dev)
    t_f = timeit(lambda: tk.attn_fwd_partial(q, k, v, crow32, col32, et32, gate, None, score="gated", norm="clampexp"))
    t_b = timeit(lambda: tk.attn_bwd_partial(q, k, v, crow32, col32, et32, gate, None, m0, dacc, dl, score="gated", norm="clampexp"))
    tiled_rows, tiled_cols = bool(tk._use_tiled(crow32, col32)), bool(tk._use_tiled_cols(col32, C))
    del q, k, v, gate, dacc, dl, m0
    # SpMM on the GCN sub-block (type-0 edges) with its weights
    g0 = et == 0
    ei0 = ei[:, g0]
    w0 = (w[g0] if w is not None else torch.rand(int(g0.sum()), device=dev, generator=gen)).to(torch.float32)
    crow0, col0, _, perm0 = build_csr(ei0, R, None)
    w0 = w0[perm0]
    a = torch.sparse_csr_tensor(crow0.to(torch.int32), col0.to(torch.int32), w0, size=(R, C))
    rows0 = torch.repeat_interleave(torch.arange(R, device=dev), crow0[1:] - crow0[:-1])
    perm_t = torch.argsort(col0, stable=True)
    ccol = torch.zeros(C + 1, dtype=torch.long, device=dev); ccol[1:] = torch.cumsum(torch.bincount(col0, minlength=C), 0)
    at = torch.sparse_csr_tensor(ccol.to(torch.int32), rows0[perm_t].to(torch.int32), w0[perm_t], size=(C, R))
    xc = torch.randn(C, d, device=dev, generator=gen); gr = torch.randn(R, d, device=dev, generator=gen)
    t_s = timeit(lambda: torch.sparse.mm(a, xc)); t_st = timeit(lambda: torch.sparse.mm(at, gr))
    E0 = int(ei0.shape[1])
    say(f"[graph]   {label:7s} R={R:9d} C={C:9d} E={E:10d} ({E/R:5.1f}/row, {E/C:5.1f}/col, rows {'tiled' if tiled_rows else 'per-row'}, "
        f"cols {'tiled' if tiled_cols else 'per-row'}): attn fwd {t_f:8.2f} bwd {t_b:8.2f} ms | spmm (E0={E0}) A x {t_s:7.2f} A^T g {t_st:7.2f} ms")
    del a, at, xc, gr, crow, col, etype, crow32, col32, et32, crow0, col0, w0, rows0, perm_t, ccol
    torch.cuda.empty_cache()
    return dict(R=R, C=C, E=E, E0=E0, attn_fwd=t_f, attn_bwd=t_b, spmm=t_s, spmmT=t_st, tiled_rows=tiled_rows, tiled_cols=tiled_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="gt3d partition directory")
    ap.add_argument("--name", required=True, help="graph name used by the model (products / reddit / ...)")
    ap.add_argument("--configs", default="4x32,8x32,2x40", help="HxD_head head configurations")
    ap.add_argument("--grids", default="", help="P_dxP_o grids whose rank-0 block is measured (comma list)")
    ap.add_argument("--blocks", default="", help="world sizes W (comma list): every divisor pair (P_d, P_o) with P_d P_o = W is measured — the candidate blocks of the 2D grids and of the head grids (P_d P_o = P / P_i)")
    ap.add_argument("--no_rand", action="store_true", help="skip the random block of the same shape (only the real-block times)")
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_retune", action="store_true", help="keep the launch configuration tuned on the first block (old behaviour)")
    args = ap.parse_args()
    global RETUNE
    RETUNE = not args.no_retune
    grids = [g for g in args.grids.split(",") if g]
    for w in [int(v) for v in args.blocks.split(",") if v]:
        grids += [f"{pd}x{w // pd}" for pd in range(1, w + 1) if w % pd == 0]
    grids = list(dict.fromkeys(grids))
    assert grids, "give --grids or --blocks"
    dev = torch.device("cuda")
    torch.cuda.set_device(0)
    say = lambda *a: print(*a, flush=True)  # noqa: E731
    meta = read_meta(args.data)
    n = meta["num_nodes"]
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    out = dict(name=args.name, data=args.data, num_nodes=n, exp_deg=args.exp_deg, grids={},
               gpu=torch.cuda.get_device_name(), retune=RETUNE)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"graph_constants_{args.name}.json")
    for gs in grids:
        pd, po = map(int, gs.split("x"))
        geom = BlockGeometry(pd, po, 0, 0, n, meta["n_pad"]).to(dev)
        t0 = time.perf_counter()
        part = load_partition(args.data, geom, dev, load_weights=True)
        gcn_blk, gcn_w = part["edge_index"], part["weight"]
        exp_blk = block_expander_edges(geom, n, args.exp_deg, seed=args.seed, device=dev)
        ei = torch.cat([gcn_blk, exp_blk], dim=1)
        et = torch.cat([torch.zeros(gcn_blk.shape[1], dtype=torch.long, device=dev), torch.ones(exp_blk.shape[1], dtype=torch.long, device=dev)])
        w = torch.cat([gcn_w.to(dev), torch.zeros(exp_blk.shape[1], device=dev)]) if gcn_w is not None else None
        del part, gcn_blk, exp_blk
        R, C, E = geom.n_row_block, geom.p_rows * geom.chunk, int(ei.shape[1])
        say(f"[graph] {args.name} grid {pd}x{po} rank 0 block: rows {R} cols {C} edges {E} (type-0 {int((et == 0).sum())}), loaded in {time.perf_counter() - t0:.0f} s")
        # the random block of the same shape and type mix
        n0 = int((et == 0).sum())
        src_r = torch.randint(0, C, (E,), device=dev, generator=gen); dst_r = torch.randint(0, R, (E,), device=dev, generator=gen)
        et_r = torch.cat([torch.zeros(n0, dtype=torch.long, device=dev), torch.ones(E - n0, dtype=torch.long, device=dev)])
        et_r = et_r[torch.randperm(E, device=dev, generator=gen)]
        ei_r = torch.stack([src_r, dst_r])
        res = {}
        for cfg in args.configs.split(","):
            h, dh = map(int, cfg.split("x"))
            real = time_block(ei, et, w, R, C, h, dh, dev, gen, f"real h{h}d{dh}", say)
            entry = dict(real=real, kv_mb=C * h * dh * 4 / 2 ** 20)
            if not args.no_rand:
                rand = time_block(ei_r, et_r, None, R, C, h, dh, dev, gen, f"rand h{h}d{dh}", say)
                kap = dict(attn_fwd=real["attn_fwd"] / rand["attn_fwd"], attn_bwd=real["attn_bwd"] / rand["attn_bwd"],
                           spmm=real["spmm"] / rand["spmm"], spmmT=real["spmmT"] / rand["spmmT"])
                say(f"[graph]   kappa h{h}d{dh}: attn fwd {kap['attn_fwd']:.3f} bwd {kap['attn_bwd']:.3f} | spmm {kap['spmm']:.3f} spmmT {kap['spmmT']:.3f}"
                    f"   (K/V footprint {C * h * dh * 4 / 2**20:.0f} MB, A footprint {E * 8 / 2**20:.0f} MB)")
                entry.update(rand=rand, kappa=kap)
            res[cfg] = entry
        out["grids"][gs] = dict(R=R, C=C, E=E, E0=n0, configs=res)
        del ei, et, w, ei_r, et_r, src_r, dst_r
        torch.cuda.empty_cache()
        with open(path, "w") as f:                       # after every grid: a time limit keeps the grids done so far
            json.dump(out, f, indent=1)
    say(f"[graph] written {path}")


if __name__ == "__main__":
    main()
