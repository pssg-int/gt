"""Single-GPU timing of one Exphormer-style attention layer (forward + backward)
on ogbn-arxiv with the official edge-list/scatter implementation vs Backend A.

  python bench/bench_attention_single_gpu.py --heads 2 --head_dim 40
  python bench/bench_attention_single_gpu.py --heads 8 --head_dim 32 --global_token

--global_token adds one virtual node connected to/from every node (degree N),
which is exactly the hub that makes the gather/scatter backward pathological.
"""
import argparse
import math
import os
import sys
import time

import torch
from torch_scatter import scatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.backend_torch import build_csr, masked_sparse_attention_torch  # noqa: E402
from gt3d.data.expander import random_regular_expander_edges  # noqa: E402
from gt3d.attention.triton_kernel import masked_sparse_attention_triton  # noqa: E402

EXPHORMER_DIR = os.environ.get("EXPHORMER_DIR", "/pscratch/sd/c/cunyang/gnn/baseline/Exphormer")


def load_arxiv_edges(device):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(EXPHORMER_DIR, "dataset"))
    data = ds[0]
    n = data.num_nodes
    ei = data.edge_index
    ei = torch.cat([ei, ei.flip(0)], dim=1)                     # add_reverse_edges
    loops = torch.arange(n).view(1, -1).repeat(2, 1)
    ei = torch.cat([ei, loops], dim=1)                          # add_self_loops (arxiv.yaml)
    return n, ei.to(device), data.x.shape[1]


def official_attention(q, k, v, gate, etype, edge_index, scale):
    """Verbatim semantics of ExphormerAttention.propagate_attention (PyG-style gather/scatter).

    The per-edge gate E is produced the way the official model does it: an nn.Embedding
    lookup by edge type (whose backward is embedding_dense_backward), not advanced indexing
    (whose backward is the sort-based index_put_ that would be pathological for 2-3 types)."""
    t, h, d = gate.shape
    gate_e = torch.nn.functional.embedding(etype, gate.reshape(t, h * d)).view(-1, h, d)
    src = k[edge_index[0]]
    dest = q[edge_index[1]]
    score = src * dest * scale
    score = score * gate_e                                       # per-edge E of shape [E,H,d]
    score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))
    msg = v[edge_index[0]] * score
    wv = torch.zeros_like(v)
    scatter(msg, edge_index[1], dim=0, out=wv, reduce="add")
    z = score.new_zeros(v.size(0), v.size(1), 1)
    scatter(score, edge_index[1], dim=0, out=z, reduce="add")
    return wv / (z + 1e-6)


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=40)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--global_token", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--impl", default="official,backend_a,triton")
    args = ap.parse_args()
    dev = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    n, ei, _ = load_arxiv_edges(dev)
    etype = torch.zeros(ei.shape[1], dtype=torch.long, device=dev)
    exp = random_regular_expander_edges(n, args.exp_deg, seed=0, device=dev)
    ei = torch.cat([ei, exp], dim=1)
    etype = torch.cat([etype, torch.ones(exp.shape[1], dtype=torch.long, device=dev)])
    n_types = 2
    if args.global_token:
        g = n
        n = n + 1
        ar = torch.arange(g, device=dev)
        gl = torch.full_like(ar, g)
        virt = torch.cat([torch.stack([ar, gl]), torch.stack([gl, ar])], dim=1)
        ei = torch.cat([ei, virt], dim=1)
        etype = torch.cat([etype, torch.full((virt.shape[1],), 2, dtype=torch.long, device=dev)])
        n_types = 3
    e = ei.shape[1]
    h, d = args.heads, args.head_dim
    scale = 1.0 / math.sqrt(d)
    print(f"arxiv: N={n} E_attn={e} (expander deg {args.exp_deg}, global_token={args.global_token}) "
          f"H={h} d_h={d} dtype={args.dtype}")

    torch.manual_seed(0)
    q = torch.randn(n, h, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(n, h, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(n, h, d, device=dev, dtype=dtype, requires_grad=True)
    gate = torch.randn(n_types, h, d, device=dev, dtype=dtype, requires_grad=True)
    crow, col, et_sorted, _ = build_csr(ei, n, etype)

    results = {}
    for impl in args.impl.split(","):
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        if impl == "official":
            def fwd():
                return official_attention(q, k, v, gate, etype, ei, scale)
        elif impl == "backend_a":
            def fwd():
                return masked_sparse_attention_torch(q, k, v, crow, col, et_sorted, gate=gate,
                                                     score="gated", norm="clampexp", scale=scale)
        elif impl == "triton":
            def fwd():
                return masked_sparse_attention_triton(q, k, v, crow, col, et_sorted, gate=gate,
                                                      score="gated", norm="clampexp", scale=scale)
        else:
            raise ValueError(impl)

        def fwd_only():
            with torch.no_grad():
                fwd()

        def fwd_bwd():
            for t in (q, k, v, gate):
                t.grad = None
            o = fwd()
            o.float().sum().backward()

        t_f = timeit(fwd_only)
        t_fb = timeit(fwd_bwd)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        results[impl] = (t_f, t_fb - t_f, peak)
        print(f"  {impl:10s} fwd {t_f*1e3:8.2f} ms   bwd {(t_fb - t_f)*1e3:8.2f} ms   peak extra mem {peak:6.2f} GB")

    if "official" in results:
        with torch.no_grad():
            o_ref = official_attention(q, k, v, gate, etype, ei, scale).float()
            for impl in results:
                if impl == "official":
                    continue
                if impl == "backend_a":
                    o = masked_sparse_attention_torch(q, k, v, crow, col, et_sorted, gate=gate,
                                                      score="gated", norm="clampexp", scale=scale)
                else:
                    o = masked_sparse_attention_triton(q, k, v, crow, col, et_sorted, gate=gate,
                                                       score="gated", norm="clampexp", scale=scale)
                print(f"  max |official - {impl}| = {(o_ref - o.float()).abs().max().item():.3e} "
                      f"(max |o| = {o_ref.abs().max().item():.3e})")

if __name__ == "__main__":
    main()
