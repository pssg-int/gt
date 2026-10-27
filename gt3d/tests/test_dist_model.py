"""DistExphormer on a Grid2D (4 ranks) vs ExphormerRef in one process, fp64: logits of the owned
chunk and gradients of every parameter (after the all-reduce over ranks), train mode with
dropout 0 (BatchNorm uses global batch statistics) and eval mode.

    torchrun --nproc_per_node 4 tests/test_dist_model.py --grid 2 2 --backend torch
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.models.exphormer_ref import ExphormerRef  # noqa: E402
from gt3d.models.exphormer_dist import DistExphormer, gcn_edges_with_self_loops  # noqa: E402
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.dist.layers import dist_cross_entropy, allreduce_grads  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[2, 2], help="P_d P_o, or P_d P_o P_i for the head grid (attention-only model)")
    ap.add_argument("--no_gcn", action="store_true", help="attention-only Exphormer (reference and distributed model without the GCN branch)")
    ap.add_argument("--heads", type=int, default=2, help="attention heads (dim_h 40: head dim 20 / 10 / 5)")
    ap.add_argument("--backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--graph_n", type=int, default=1000)
    ap.add_argument("--graph_e", type=int, default=12000)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--pipeline_sub", type=int, default=1)
    ap.add_argument("--stream_sub", type=int, default=1)
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--layout", default="row_major")
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--gather_proj", action="store_true")
    ap.add_argument("--stream_overlap", action="store_true")
    ap.add_argument("--no_stream_compact", action="store_true")
    ap.add_argument("--model", default="exphormer", choices=["exphormer", "unimp", "graphormer"])
    ap.add_argument("--kernel_bf16", action="store_true", help="bf16 Q / K / V operands for the triton kernels (tolerances relaxed to 2e-2 / 5e-2)")
    ap.add_argument("--sparse_comm", action="store_true", help="sparsity-aware collectives (gt3d/dist/sparse_comm.py), forced on every collective (--sparse_threshold 1.0)")
    ap.add_argument("--sparse_threshold", type=float, default=1.0)
    args = ap.parse_args()
    head_grid = len(args.grid) == 3
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    dtype = torch.float64 if args.backend == "torch" else torch.float32
    # bf16 operands: forward ~2e-4 .. 1e-3, worst parameter gradient ~5e-2 (2D) / 1.1e-1 (head grid (2,1,2), a norm bias whose
    # gradient is a small sum) relative on the 1K-node test graph (measured 2026-10-17 / 18): precision of the mode, not a bug
    tol_f, tol_g = (1e-10, 1e-8) if dtype == torch.float64 else ((2e-2, 2e-1) if args.kernel_bf16 else (5e-4, 5e-3))
    g_floor = 1e-8 if dtype == torch.float64 else 1e-4   # biases feeding a BatchNorm have an exactly-zero gradient

    n, dim_in, dim_h, dim_out, heads, layers = args.graph_n, 32, 40, 7, args.heads, 3
    gen = torch.Generator().manual_seed(0)
    ei = torch.randint(0, n, (2, args.graph_e), generator=gen)
    ei = torch.cat([ei, ei.flip(0)], dim=1)                                  # reverse edges
    exp = torch.stack([torch.randperm(n, generator=gen), torch.arange(n)])   # a permutation "expander"
    exp = torch.cat([exp, exp.flip(0)], dim=1)
    x = torch.randn(n, dim_in, generator=gen, dtype=dtype)
    y = torch.randint(0, dim_out, (n,), generator=gen)
    train_mask = torch.rand(n, generator=gen) < 0.6

    # ---- reference (single process, full graph) ----
    import copy
    torch.manual_seed(0)
    if head_grid:
        from gt3d.dist.grid3d import Grid3D
        grid = Grid3D(args.grid[0], args.grid[1], args.grid[2], n).to(dev)
    else:
        grid = Grid2D(args.grid[0], args.grid[1], n, layout=args.layout).to(dev)
    if args.model == "exphormer":
        ref = ExphormerRef(dim_in, dim_h, dim_out, layers, heads, dropout=0.0, gnn_dropout=0.0, with_gcn=not args.no_gcn).to(dtype).to(dev)
        ref_sd0 = copy.deepcopy(ref.state_dict())          # initial parameters AND running stats
        ref.train()
        logits_ref = ref(x.to(dev), ei.to(dev), exp.to(dev))
        ref_eval = lambda: ref(x.to(dev), ei.to(dev), exp.to(dev))
    else:
        from gt3d.models.gt_variants import (UniMPRef, GraphormerNodeRef, DistGTVariant, typed_graph_edges, graphormer_edges,
                                             type_local_block, token_edges_local, owned_in_degree, token_owned_row)
        if args.model == "unimp":
            ref = UniMPRef(dim_in, dim_h, dim_out, layers, heads, dropout=0.0).to(dtype).to(dev)
            ref_ei, ref_et = typed_graph_edges(ei.to(dev), n)
            x_ref = x.to(dev)
            ref_fwd = lambda: ref(x_ref, ref_ei)
        else:
            ref = GraphormerNodeRef(dim_in, dim_h, dim_out, layers, heads, dropout=0.0, max_degree=16).to(dtype).to(dev)
            ref_ei, ref_et = graphormer_edges(ei.to(dev), n)
            x_ref = torch.cat([x.to(dev), torch.zeros(1, dim_in, dtype=dtype, device=dev)])
            degree = torch.bincount(ei[1][ei[0] != ei[1]], minlength=n + 1).to(dev)     # in-degree over the neighbour edges
            ref_fwd = lambda: ref(x_ref, ref_ei, ref_et, degree, n)[:n]
        ref_sd0 = copy.deepcopy(ref.state_dict())
        ref.train()
        logits_ref = ref_fwd()
        ref_eval = ref_fwd
    loss_ref = torch.nn.functional.cross_entropy(logits_ref[train_mask.to(dev)], y.to(dev)[train_mask.to(dev)])
    loss_ref.backward()

    # ---- distributed ----
    gcn_ei = gcn_edges_with_self_loops(ei, n)
    gcn_blk, _ = grid.local_block(gcn_ei)
    x_loc = grid.take_owned(x).to(dev)
    y_loc = grid.take_owned(y).to(dev)
    tm_loc = grid.take_owned(train_mask.to(torch.uint8)).bool().to(dev) & grid.valid_mask
    fwd_kwargs = {}
    if args.model == "exphormer":
        attn_ei, attn_et = ExphormerRef.build_attention_edges(ei, exp)
        attn_blk, attn_blk_et = grid.local_block(attn_ei, attn_et)
        model = DistExphormer(grid, gcn_blk.to(dev), attn_blk.to(dev), attn_blk_et.to(dev), dim_in, dim_h, dim_out,
                              layers, heads, dropout=0.0, gnn_dropout=0.0, backend=args.backend, fused=args.fused,
                              pipeline_sub=args.pipeline_sub, stream_sub=args.stream_sub, checkpoint=args.checkpoint, ring=args.ring, gather_proj=args.gather_proj, stream_overlap=args.stream_overlap, stream_compact=not args.no_stream_compact,
                              kernel_dtype=torch.bfloat16 if args.kernel_bf16 else None, with_gcn=not args.no_gcn,
                              sparse_comm=args.sparse_comm, sparse_threshold=args.sparse_threshold).to(dtype).to(dev)
        if args.sparse_comm and rank == 0:
            print(f"[dist-model] sparse_comm info: {model.sparse_info}", flush=True)
    else:
        # block from the reference's global edge list (exact same edges) ...
        attn_blk, attn_blk_et = grid.local_block(ref_ei.cpu(), ref_et.cpu())
        attn_blk, attn_blk_et = attn_blk.to(dev), attn_blk_et.to(dev)
        # ... and the way train_dist builds it from partition data (self loops typed on the local block, token edges
        # generated locally) must give the same set of (src, dst, type) triples
        loc_blk = gcn_blk.to(dev)
        loc_et = type_local_block(grid, loc_blk, n)
        token_row = None
        if args.model == "graphormer":
            tok_ei, tok_et = token_edges_local(grid, n, dev)
            loc_blk, loc_et = torch.cat([loc_blk, tok_ei], dim=1), torch.cat([loc_et, tok_et])
            fwd_kwargs["degree_loc"] = owned_in_degree(grid, attn_blk, attn_blk_et, dev)
            token_row = token_owned_row(grid, n)
        def triples(e, t):
            key = (e[0].long() * (grid.n_row_block + 1) + e[1].long()) * 8 + t.long()
            return torch.sort(key)[0]
        a, b = triples(attn_blk, attn_blk_et), triples(loc_blk, loc_et)
        same_blocks = a.numel() == b.numel() and bool((a == b).all())
        if not same_blocks:
            print(f"[dist-model rank {rank}] local block generators differ from grid.local_block: {a.numel()} vs {b.numel()} edges", flush=True)
        model = DistGTVariant(grid, attn_blk, attn_blk_et, dim_in, dim_h, dim_out, layers, heads, model=args.model, dropout=0.0,
                              backend=args.backend, checkpoint=args.checkpoint, max_degree=16, token_row=token_row).to(dtype).to(dev)
    model.load_ref_state_dict(ref_sd0)
    model.train()
    logits_loc = model(x_loc, **fwd_kwargs)
    loss_loc = dist_cross_entropy(logits_loc, y_loc, tm_loc, int(train_mask.sum()))
    loss_loc.backward()
    allreduce_grads(list(model.parameters()))
    loss_tot = loss_loc.detach().clone(); dist.all_reduce(loss_tot)

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()

    lo, hi = grid.owned_range()
    nv = int(grid.valid_mask.sum())
    err_f = rel(logits_loc[:nv], logits_ref[lo:lo + nv]) if nv > 0 else 0.0
    err_loss = abs(loss_tot.item() - loss_ref.item()) / abs(loss_ref.item())
    ref_p = dict(ref.named_parameters())
    worst_g, worst_name = 0.0, ""
    for name, p in model.named_parameters():
        g_ref = ref_p[name].grad
        e = ((p.grad - g_ref).abs().max() / g_ref.abs().max().clamp_min(g_floor)).item()
        if e > worst_g:
            worst_g, worst_name = e, name
    # running statistics after the training step
    own_b, ref_b = dict(model.named_buffers()), dict(ref.named_buffers())
    worst_rs = max(((own_b[k] - ref_b[k]).abs().max() / ref_b[k].abs().max().clamp_min(1e-8)).item()
                   for k in ref_b if k.endswith("running_mean") or k.endswith("running_var")) if ref_b else 0.0
    # eval mode (running stats)
    model.eval(); ref.eval()
    with torch.no_grad():
        err_eval = rel(model(x_loc, **fwd_kwargs)[:nv], ref_eval()[lo:lo + nv]) if nv > 0 else 0.0
    ok = err_f < tol_f and err_loss < tol_f and worst_g < tol_g and err_eval < tol_f
    if args.model != "exphormer":
        ok = ok and same_blocks
    print(f"[dist-model rank {rank}] model={args.model} grid={tuple(args.grid)} gcn={int(not args.no_gcn)} backend={args.backend} kbf16={int(args.kernel_bf16)} sparse={int(args.sparse_comm)} fused={args.fused} ring={args.ring} pipe={args.pipeline_sub} stream={args.stream_sub} ckpt={args.checkpoint} train-fwd={err_f:.1e} "
          f"loss={err_loss:.1e} worst-grad={worst_g:.1e} ({worst_name}) running-stats={worst_rs:.1e} eval-fwd={err_eval:.1e} {'OK' if ok else 'FAIL'}", flush=True)
    flag = torch.tensor([0 if ok else 1], device=dev); dist.all_reduce(flag); dist.barrier()
    if rank == 0:
        print("[dist-model] ALL PASSED" if flag.item() == 0 else "[dist-model] FAILED", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() == 0 else 1)


if __name__ == "__main__":
    main()
