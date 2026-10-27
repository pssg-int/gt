"""E1: one structure-masked sparse attention layer (Exphormer semantics: gated scores, clamp-exp normalisation, edge
types) on real graph blocks, single GPU, forward and backward time and peak memory, ours vs external implementations:

  official   PyG-style gather / scatter of the official Exphormer code ([E, H, d] messages, scatter-add)
  dgl        DGL fused message passing: gsddmm (u_dot_v) + edge-wise exp + gspmm (copy_e_sum, u_mul_e_sum); [E, H] scores
  cusparse   library pipeline (DAC'26-style): cuSPARSE SDDMM (torch.sparse.sampled_addmm) per head and edge type,
             segment softmax on the CSR values, cuSPARSE SpMM (torch.sparse.mm); [E] scores per head
  backend_a  ours without Triton: CSR + torch_scatter segment_csr, explicit backward (no index_put)
  triton     ours: fused per-row / tiled Triton kernels (4 heads per program, row + column backward passes)
  fa2_dense  FlashAttention-2 dense attention on the same rows (no mask: the cost floor of every dense / block-dense
             approach incl. FlexAttention, whose block mask on a randomly ordered graph has no empty 128x128 block);
             only on the 64K-node arxiv subset

Graphs: arxiv64k (first 64K nodes, induced), arxiv (full, + expander), products / Reddit grid blocks of the 4x4 and 2x2
grids (rank (0, 0) of the partition, + self loops already in the partition, + random expander edges with the block's
share 6 / P_o per row), optionally a global token (one node connected to / from every row: the hub).

    python bench/bench_kernel_external.py --graphs arxiv64k,arxiv,reddit4,products16,products4 --configs 2x40,4x32,8x32
Prints one line per (graph, config, implementation): fwd ms, bwd ms, peak extra memory GB, max |o - official|.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
from torch_scatter import scatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.backend_torch import build_csr, masked_sparse_attention_torch  # noqa: E402
from gt3d.attention.triton_kernel import masked_sparse_attention_triton  # noqa: E402
from gt3d.data.expander import random_regular_expander_edges  # noqa: E402

EXPHORMER_DIR = os.environ.get("EXPHORMER_DIR", "/pscratch/sd/c/cunyang/gnn/baseline/Exphormer")
DATA = "/pscratch/sd/c/cunyang/gnn/gt/data"
EPS = 1e-6


# ---------------------------------------------------------------------------------------------------- graphs
def load_arxiv(dev, subset=None):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(EXPHORMER_DIR, "dataset"))
    data = ds[0]
    n = data.num_nodes
    ei = data.edge_index
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    if subset is not None:
        keep = (ei[0] < subset) & (ei[1] < subset)
        ei, n = ei[:, keep], subset
    loops = torch.arange(n).view(1, -1).repeat(2, 1)
    ei = torch.cat([ei, loops], dim=1).to(dev)
    et = torch.zeros(ei.shape[1], dtype=torch.long, device=dev)
    exp = random_regular_expander_edges(n, 3, seed=0, device=dev)
    ei = torch.cat([ei, exp], dim=1)
    et = torch.cat([et, torch.ones(exp.shape[1], dtype=torch.long, device=dev)])
    return n, n, ei, et


def load_block(name, P, dev):
    """Rank (0, 0) of the P x P grid (P = p^2) of a gt3d partition: a square block (rows = row block 0, columns =
    column block 0), edges + the block's share of random expander edges (6 / p per row, type 1)."""
    part = f"{DATA}/{name}"
    meta = json.load(open(f"{part}/meta.json"))
    p = int(round(math.sqrt(P)))
    assert p * p == P and meta["p_part"] % P == 0, (P, meta["p_part"])
    n_pad, pc = meta["n_pad"], meta["chunk"]
    gc = n_pad // P                                            # grid chunk
    row_ranges = [(c * p * gc, (c * p + 1) * gc) for c in range(p)]       # grid chunks {c' P_d + 0}
    col_range = (0, p * gc)                                    # grid chunks 0 .. P_d - 1 (contiguous)
    src_chunks = range(0, col_range[1] // pc)
    dst_chunks = sorted({i for lo, hi in row_ranges for i in range(lo // pc, (hi - 1) // pc + 1)})
    eis = []
    for i in src_chunks:
        for j in dst_chunks:
            ei, _ = torch.load(f"{part}/blocks/{i}_{j}.pt", weights_only=False)
            eis.append(ei)
    ei = torch.cat(eis, dim=1)
    src, dst = ei[0], ei[1]
    keep = (src >= col_range[0]) & (src < col_range[1])
    dst_local = torch.full_like(dst, -1)
    for c, (lo, hi) in enumerate(row_ranges):
        m = (dst >= lo) & (dst < hi)
        dst_local[m] = c * gc + (dst[m] - lo)
    keep &= dst_local >= 0
    src_l, dst_l = src[keep].to(dev), dst_local[keep].to(dev)
    R, C = p * gc, p * gc
    et = torch.zeros(src_l.numel(), dtype=torch.long, device=dev)
    # expander share: 6 / p edges per row, random sources in the column block
    gen = torch.Generator(device=dev).manual_seed(0)
    n_exp = max(1, 6 // p)
    e_dst = torch.arange(R, device=dev).repeat_interleave(n_exp)
    e_src = torch.randint(0, C, (e_dst.numel(),), device=dev, generator=gen)
    ei = torch.stack([torch.cat([src_l, e_src]), torch.cat([dst_l, e_dst])])
    et = torch.cat([et, torch.ones(e_dst.numel(), dtype=torch.long, device=dev)])
    return R, C, ei, et


def add_global_token(R, C, ei, et, dev):
    """One extra node (index R == C) connected to and from every row / column (edge type 2)."""
    assert R == C
    ar = torch.arange(R, device=dev)
    g = torch.full_like(ar, R)
    virt = torch.cat([torch.stack([ar, g]), torch.stack([g, ar])], dim=1)
    ei = torch.cat([ei, virt], dim=1)
    et = torch.cat([et, torch.full((virt.shape[1],), 2, dtype=torch.long, device=dev)])
    return R + 1, C + 1, ei, et


# ---------------------------------------------------------------------------------------------------- implementations
def official_attention(q, k, v, gate, etype, ei, scale):
    """ExphormerAttention.propagate_attention (PyG gather / scatter), gate via embedding lookup as in the official code."""
    t, h, d = gate.shape
    gate_e = torch.nn.functional.embedding(etype, gate.reshape(t, h * d)).view(-1, h, d)
    score = (k[ei[0]] * q[ei[1]] * scale * gate_e).sum(-1, keepdim=True)
    score = torch.exp(score.clamp(-5, 5))
    msg = v[ei[0]] * score
    wv = torch.zeros_like(q)
    scatter(msg, ei[1], dim=0, out=wv, reduce="add")
    z = score.new_zeros(q.size(0), q.size(1), 1)
    scatter(score, ei[1], dim=0, out=z, reduce="add")
    return wv / (z + EPS)


class DGLImpl:
    def __init__(self, R, C, ei, et, dev):
        import dgl
        self.dgl = dgl
        n = max(R, C)
        self.g = dgl.graph((ei[0], ei[1]), num_nodes=n, device=dev)
        self.et = et
        self.R, self.C, self.n = R, C, n

    def __call__(self, q, k, v, gate, scale):
        ops = self.dgl.ops
        n = self.n
        qq = torch.cat([q, q.new_zeros(n - q.shape[0], *q.shape[1:])]) if q.shape[0] < n else q
        kk = torch.cat([k, k.new_zeros(n - k.shape[0], *k.shape[1:])]) if k.shape[0] < n else k
        vv = torch.cat([v, v.new_zeros(n - v.shape[0], *v.shape[1:])]) if v.shape[0] < n else v
        s = None
        for t in range(gate.shape[0]):                       # per-type gated dot product: q . (k * g_t)
            s_t = ops.u_dot_v(self.g, kk * gate[t], qq) * scale          # [E, H, 1]
            s = s_t if s is None else torch.where((self.et == t).view(-1, 1, 1), s_t, s)
        p = torch.exp(s.clamp(-5, 5))
        z = ops.copy_e_sum(self.g, p)                         # [n, H, 1]
        o = ops.u_mul_e_sum(self.g, vv, p)                    # [n, H, d]
        return (o / (z + EPS))[: self.R]


class CusparseImpl:
    def __init__(self, R, C, crow, col, et, dev):
        self.crow, self.col = crow.to(torch.int64).contiguous(), col.to(torch.int64).contiguous()
        self.et = et
        self.R, self.C = R, C
        self.mask = torch.sparse_csr_tensor(self.crow, self.col, torch.ones(col.numel(), device=dev), size=(R, C))
        self.ones = torch.ones(C, 1, device=dev)

    def __call__(self, q, k, v, gate, scale):
        outs = []
        for h in range(q.shape[1]):
            qh, kh, vh = q[:, h].contiguous(), k[:, h].contiguous(), v[:, h].contiguous()
            s = None
            for t in range(gate.shape[0]):
                s_t = torch.sparse.sampled_addmm(self.mask, qh * scale, (kh * gate[t, h]).t().contiguous(), beta=0.0).values()
                s = s_t if s is None else torch.where(self.et == t, s_t, s)
            p = torch.exp(s.clamp(-5, 5))
            P = torch.sparse_csr_tensor(self.crow, self.col, p, size=(self.R, self.C))
            z = torch.sparse.mm(P, self.ones)
            outs.append(torch.sparse.mm(P, vh) / (z + EPS))
        return torch.stack(outs, dim=1)


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="arxiv64k,arxiv,reddit4,products16,products4")
    ap.add_argument("--configs", default="2x40,4x32,8x32")
    ap.add_argument("--impls", default="official,dgl,cusparse,backend_a,triton,fa2_dense")
    ap.add_argument("--global_token", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda")
    say = lambda *a: print(*a, flush=True)
    for gname in args.graphs.split(","):
        if gname == "arxiv64k":
            R, C, ei, et = load_arxiv(dev, subset=65536)
        elif gname == "arxiv":
            R, C, ei, et = load_arxiv(dev)
        elif gname.startswith("products") or gname.startswith("reddit"):
            P = int(gname.replace("products", "").replace("reddit", ""))
            R, C, ei, et = load_block("products_p16b" if gname.startswith("products") else "reddit_p16", P, dev)
        else:
            raise ValueError(gname)
        if args.global_token:
            R, C, ei, et = add_global_token(R, C, ei, et, dev)
        E = ei.shape[1]
        n_types = int(et.max().item()) + 1
        crow, col, et_sorted, _ = build_csr(ei, R, et)
        col32, et32 = col.to(torch.int32), et_sorted.to(torch.int32)
        say(f"=== {gname}{' +global_token' if args.global_token else ''}: rows {R} cols {C} edges {E} ({E / R:.1f} per row), types {n_types}")
        for cfg in args.configs.split(","):
            h, d = (int(x) for x in cfg.split("x"))
            scale = 1.0 / math.sqrt(d)
            torch.manual_seed(0)
            q = torch.randn(R, h, d, device=dev, requires_grad=True)
            k = torch.randn(C, h, d, device=dev, requires_grad=True)
            v = torch.randn(C, h, d, device=dev, requires_grad=True)
            gate = torch.randn(n_types, h, d, device=dev, requires_grad=True)
            o_ref = None
            for impl in args.impls.split(","):
                if impl == "fa2_dense" and gname != "arxiv64k":
                    continue
                try:
                    if impl == "official":
                        fwd = lambda: official_attention(q, k, v, gate, et, ei, scale)
                    elif impl == "dgl":
                        obj = DGLImpl(R, C, ei, et, dev)
                        fwd = lambda: obj(q, k, v, gate, scale)
                    elif impl == "cusparse":
                        obj = CusparseImpl(R, C, crow, col, et_sorted, dev)
                        fwd = lambda: obj(q, k, v, gate, scale)
                    elif impl == "backend_a":
                        fwd = lambda: masked_sparse_attention_torch(q, k, v, crow, col, et_sorted, gate=gate, score="gated", norm="clampexp", scale=scale)
                    elif impl == "triton":
                        fwd = lambda: masked_sparse_attention_triton(q, k, v, crow, col32, et32, gate=gate, score="gated", norm="clampexp", scale=scale)
                    elif impl == "fa2_dense":
                        from flash_attn import flash_attn_func
                        qb, kb, vb = (t.to(torch.bfloat16).unsqueeze(0) for t in (q, k, v))
                        fwd = lambda: flash_attn_func(qb, kb, vb, causal=False).squeeze(0)
                    else:
                        raise ValueError(impl)
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()

                    def fwd_only():
                        with torch.no_grad():
                            fwd()

                    def fwd_bwd():
                        for t_ in (q, k, v, gate):
                            t_.grad = None
                        fwd().float().sum().backward()
                    t_f = timeit(fwd_only, iters=args.iters)
                    try:
                        t_fb = timeit(fwd_bwd, iters=args.iters)
                        bwd = f"{t_fb - t_f:8.2f}"
                    except Exception as ex:                      # e.g. no backward for CSR sparse.mm w.r.t. the values
                        bwd = "   n/a  "
                        msg = str(ex).splitlines()[0][:70]
                    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
                    with torch.no_grad():
                        o = fwd().float()
                    if impl == "official":
                        o_ref = o
                    err = "" if (o_ref is None or impl == "fa2_dense") else f"  max|o-official| {(o - o_ref).abs().max().item():.2e}"
                    say(f"  {cfg:5s} {impl:10s} fwd {t_f:8.2f} ms  bwd {bwd} ms  peak extra {peak:6.2f} GB{err}" + (f"  ({msg})" if bwd.strip() == "n/a" else ""))
                    del o
                except torch.OutOfMemoryError:
                    say(f"  {cfg:5s} {impl:10s} OOM")
                except Exception as ex:
                    say(f"  {cfg:5s} {impl:10s} FAILED: {str(ex).splitlines()[0][:120]}")
                    traceback.print_exc(limit=1)
                for t_ in (q, k, v, gate):
                    t_.grad = None
                torch.cuda.empty_cache()
            del q, k, v, gate
            torch.cuda.empty_cache()
        del ei, et, crow, col
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
