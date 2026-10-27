"""DistGraphormerNode (TorchGT's node-level Graphormer on the 2D grid), fused-communication blocks vs the
baseline blocks vs the single-process GraphormerNode: logits of the owned chunk and parameter gradients
(after the all-reduce), dropout 0.

    torchrun --nproc_per_node 4 tests/test_graphormer_fused.py --grid 2 2 --backend torch     (fp64, fused vs baseline)
    torchrun --nproc_per_node 4 tests/test_graphormer_fused.py --grid 2 2 --backend triton    (fp32, both vs single GPU)
"""
import argparse
import copy
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.models.graphormer_node import GraphormerNode, DistGraphormerNode  # noqa: E402
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.dist.layers import dist_cross_entropy, allreduce_grads  # noqa: E402


def run_dist(model, grid, xg, yg, mask, n):
    params = list(model.parameters())
    out = model(xg)
    loss = dist_cross_entropy(out, yg, mask, n - 1)
    loss.backward()
    allreduce_grads(params)
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
    return out.detach(), grads


def compare(name, a, b, tol, ok, floor=1e-12):
    err = (a - b).abs().max().item()
    ref = b.abs().max().item()
    rel = err / max(ref, 1e-30)
    good = rel < tol or err < floor                 # floor: the K bias has a mathematically zero gradient (row-constant scores)
    print(f"[rank {dist.get_rank()}] {name}: max abs {err:.3e} rel {rel:.3e} {'ok' if good else 'FAIL'}", flush=True)
    return ok and good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    ap.add_argument("--backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--graph_n", type=int, default=1500)
    ap.add_argument("--graph_e", type=int, default=15000)
    ap.add_argument("--comm_bf16", action="store_true")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    dtype = torch.float64 if args.backend == "torch" else torch.float32
    tol = 1e-9 if dtype == torch.float64 else 2e-3
    floor = 1e-12 if dtype == torch.float64 else 1e-8
    if args.comm_bf16:
        tol = 5e-2

    n, dim_in, hidden, ffn, heads, layers, ncls = args.graph_n, 16, 32, 32, 4, 2, 5
    gen = torch.Generator().manual_seed(0)
    ei = torch.randint(1, n, (2, args.graph_e), generator=gen)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    nodes = torch.arange(1, n)
    glob = torch.stack([torch.zeros_like(nodes), nodes])
    ei = torch.cat([ei, glob, glob.flip(0)], dim=1)                       # global token (node 0) <-> every node
    x = torch.randn(n - 1, dim_in, generator=gen, dtype=dtype)
    y = torch.randint(0, ncls, (n - 1,), generator=gen)

    torch.manual_seed(0)
    ref = GraphormerNode(dim_in, hidden, ffn, heads, layers, ncls, ei.to(dev), n, dropout=0.0, input_dropout=0.0).to(dtype).to(dev)
    sd0 = copy.deepcopy(ref.state_dict())
    ref.train()
    ok = True
    if args.backend == "triton":                                           # single-process reference (Triton kernel, fp32)
        out_ref = ref(x.to(dev))
        loss = torch.nn.functional.cross_entropy(out_ref, y.to(dev))
        loss.backward()
        grads_ref = {k: p.grad.detach().clone() for k, p in ref.named_parameters()}

    grid = Grid2D(args.grid[0], args.grid[1], n).to(dev)
    blk, _ = grid.local_block(ei)
    xfull = torch.cat([torch.zeros(1, dim_in, dtype=dtype), x]); yfull = torch.cat([torch.zeros(1, dtype=torch.long), y])
    xg, yg = grid.take_owned(xfull).to(dev), grid.take_owned(yfull).to(dev)
    mask = grid.valid_mask.clone(); lo, hi = grid.owned_range()
    if lo == 0:
        mask[0] = False
    cdt = torch.bfloat16 if args.comm_bf16 else None
    results = {}
    for fused in (False, True):
        torch.manual_seed(0)
        m = DistGraphormerNode(grid, (blk.to(dev), None), dim_in, hidden, ffn, heads, layers, ncls, dropout=0.0,
                               input_dropout=0.0, backend=args.backend, comm_dtype=cdt, fused=fused).to(dtype).to(dev)
        missing, unexpected = m.load_state_dict(sd0, strict=False)
        assert not unexpected and not missing, (missing, unexpected)
        m.train()
        results[fused] = run_dist(m, grid, xg, yg, mask, n)
    out_a, g_a = results[False]
    out_b, g_b = results[True]
    ok = compare("logits fused vs baseline", out_b[mask], out_a[mask], tol, ok, floor)
    for k in g_a:
        ok = compare(f"grad {k} fused vs baseline", g_b[k], g_a[k], tol, ok, floor)
    if args.backend == "triton":
        owned = torch.arange(lo, hi, device=dev)[mask]                       # global ids of the owned real nodes
        ref_rows = out_ref.detach()[owned - 1]
        for fused, (o, g) in results.items():
            tag = "fused" if fused else "baseline"
            ok = compare(f"logits {tag} vs single GPU", o[mask], ref_rows, tol, ok, floor)
            for k in g:
                ok = compare(f"grad {k} {tag} vs single GPU", g[k], grads_ref[k], tol, ok, floor)
    flags = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(flags)
    if rank == 0:
        print("[graphormer-fused] ALL PASSED" if flags.item() == 0 else "[graphormer-fused] FAILED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
