"""Like-for-like kernel comparison with the DAC'26 stack: one Sparse Graph Attention (paper Eq. 4-5: dot-product scores
on the graph's edges, row softmax, weighted sum of V; h heads) forward + backward on the WHOLE graph, one GPU, fp32,
with our fused Triton kernel (score="dot", norm="softmax").  The DGL side (bsddmm -> softmax -> bspmm, DGL 2.3) is
`baseline/dac26/dac26_sga.py --kernel`; both read the same edge lists (baseline/dac26_graphs, prepare_graphs.py).

    python bench/bench_sga_kernel.py --graphs proteins reddit products --heads 8 --head_dim 16
Prints one RESULT line per graph (same fields as the DGL side).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.backend_torch import build_csr  # noqa: E402
from gt3d.attention.triton_kernel import masked_sparse_attention_triton  # noqa: E402

GRAPHS = "/pscratch/sd/c/cunyang/gnn/gt/baseline/dac26_graphs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["proteins", "reddit", "products"])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--check_dir", default=None, help="compare one call with the DGL side's --dump files <dir>/<graph>.pt")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    for name in args.graphs:
        gr = torch.load(os.path.join(GRAPHS, f"{name}.pt"))
        n, ei = gr["num_nodes"], gr["edge_index"].to(dev)
        crow, col, _, _ = build_csr(ei, n)
        e = int(ei.shape[1])
        del ei, gr
        h, dh = args.heads, args.head_dim
        g = torch.Generator(device="cpu").manual_seed(0)
        q, k, v = (torch.randn(n, h, dh, generator=g).to(dev).requires_grad_() for _ in range(3))
        dout = torch.randn(n, h, dh, generator=g).to(dev)

        def fwd():
            return masked_sparse_attention_triton(q, k, v, crow, col, score="dot", norm="softmax")

        ref_path = os.path.join(args.check_dir, f"{name}.pt") if args.check_dir else None
        if ref_path and os.path.exists(ref_path):                              # same inputs as the DGL side's --dump
            ref = torch.load(ref_path)
            out = fwd(); out.backward(dout)
            errs = {key: ((a.detach().cpu() - ref[key]).abs().max() / ref[key].abs().max()).item()
                    for key, a in (("out", out), ("dq", q.grad), ("dk", k.grad), ("dv", v.grad))}
            print(f"CHECK graph={name} vs DGL: " + " ".join(f"{key}_rel_err={v_:.1e}" for key, v_ in errs.items()), flush=True)
            q.grad = k.grad = v.grad = None
            del out, ref
        for _ in range(3):
            fwd().backward(dout)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        s, m, en = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        f_ms = b_ms = 0.0
        for _ in range(args.iters):
            s.record(); out = fwd(); m.record(); out.backward(dout); en.record()
            torch.cuda.synchronize()
            f_ms += s.elapsed_time(m); b_ms += m.elapsed_time(en)
            del out
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        print(f"RESULT kernel impl=ours graph={name} N={n} E={e} heads={h} head_dim={dh} "
              f"fwd_ms={f_ms / args.iters:.2f} bwd_ms={b_ms / args.iters:.2f} peak_extra_gb={peak:.2f}", flush=True)
        del q, k, v, dout, crow, col
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
