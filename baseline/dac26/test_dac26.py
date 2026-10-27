"""Parity of the DGL re-implementation: GP-AG and GP-A2A on p ranks against the single-GPU SGA model on a random graph
(outputs of the owned rows and all weight gradients, fp64).  One process per GPU (srun -n p):

    python test_dac26.py --strategy gp_ag ; python test_dac26.py --strategy gp_a2a
"""
import argparse
import os

import torch
import torch.distributed as dist
import dgl.sparse as dglsp

from dac26_sga import Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["gp_ag", "gp_a2a"])
    ap.add_argument("--num_nodes", type=int, default=1001)                     # not a multiple of p: padding path
    ap.add_argument("--num_edges", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, p = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dev = torch.device("cuda")
    dt = torch.float64
    n, e = args.num_nodes, args.num_edges
    g = torch.Generator().manual_seed(0)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = torch.unique(torch.cat([ei, ei.flip(0)], 1), dim=1)                   # symmetric, no duplicates
    n_pad = -(-n // p) * p
    chunk = n_pad // p
    x_all = torch.zeros(n_pad, args.hidden, dtype=dt)
    x_all[:n] = torch.randn(n, args.hidden, generator=g, dtype=dt)

    torch.manual_seed(1)
    ref = Net(args.hidden, args.heads, 2, 5, "single", 1).to(dev, dt)
    torch.manual_seed(1)
    mod = Net(args.hidden, args.heads, 2, 5, args.strategy, p).to(dev, dt)
    A_full = dglsp.spmatrix(torch.stack([ei[1], ei[0]]).to(dev), shape=(n_pad, n_pad))
    out_ref = ref(A_full, x_all.to(dev))
    w = torch.randn(out_ref.shape, generator=g, dtype=dt).to(dev)
    (out_ref * w).sum().backward()

    lo, hi = rank * chunk, (rank + 1) * chunk
    if args.strategy == "gp_ag":
        m = (ei[1] >= lo) & (ei[1] < hi)
        A = dglsp.spmatrix(torch.stack([ei[1][m] - lo, ei[0][m]]).to(dev), shape=(chunk, n_pad))
    else:
        A = A_full
    out = mod(A, x_all[lo:hi].to(dev))
    (out * w[lo:hi]).sum().backward()
    err_out = (out - out_ref[lo:hi]).abs().max().item()
    err_grad = 0.0
    for a, b in zip(mod.parameters(), ref.parameters()):
        gsum = a.grad.clone(); dist.all_reduce(gsum)
        err_grad = max(err_grad, ((gsum - b.grad).abs().max() / b.grad.abs().max()).item())
    errs = torch.tensor([err_out, err_grad], device=dev)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    if rank == 0:
        ok = errs[0] < 1e-10 and errs[1] < 1e-10
        print(f"TEST {args.strategy} p={p} N={n} E={ei.shape[1]} heads={args.heads}: max|out err|={errs[0]:.2e} "
              f"max rel grad err={errs[1]:.2e} {'PASSED' if ok else 'FAILED'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
