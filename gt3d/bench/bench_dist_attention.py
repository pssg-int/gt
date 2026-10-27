"""Attention-only distributed benchmark: forward + backward of ONE structure-masked sparse attention layer
(Q/K/V projections on the gathered / rotated blocks, communication, fused kernel, combine) for the context-parallel
strategies.  No GCN branch, FFN, norms, dropout, loss or optimizer: this is the "pure GT" cost of a strategy.

    strategy = gather : the fused layer's attention path on the given grid: all-gather x over the column group
                        (K, V side) and the row group (Q side), block partials, reduce-scatter combine.
                        grid (P, 1) = DAC'26 GP-AG (all-gather of the whole K/V side); 2D grid = ours.
    strategy = gather_proj : the same grid and collectives, but the OWNER projects its chunk once and the projections
                        are gathered instead of x: Q over the row group (width d, same bytes as x), [K | V] over the
                        column group (width 2d, twice the bytes of x); no redundant K/V projection of the column block
                        (compute traded for bytes).
    strategy = ring   : ring attention on a (P, 1) grid (gt3d/dist/ring.py), GCN branch disabled: the layer INPUT x
                        rotates (P-1 hops of chunk x hidden), K/V projected per hop and recomputed in backward.
    strategy = ring_kv: Ring Attention as published (gt3d/dist/ring_kv.py): the owner projects K/V once and the packed
                        [K | V] chunk rotates (P-1 hops of 2 x chunk x hidden), the dK/dV accumulator rotates in backward.
    strategy = ring_kv2: the same with a bidirectional ring (half of every chunk's rows clockwise, half counter-clockwise:
                        both directions of every link, same bytes per rank, one kernel per step).
    strategy = head   : Grid3D (P_d, P_o, P_i) with a DeepSpeed-Ulysses head axis (gt3d/dist/grid3d.py +
                        DistHeadSparseAttention): owners project Q/K/V, all-to-all rows <-> heads inside the head
                        group, all-gather of the head slices over the column / row groups, block kernel on H/P_i
                        heads, reduce-scatter combine, all-to-all back.  (P_d, P_o, 1) = 2D with projected gathers;
                        (1, 1, P) = pure head parallelism (GP-A2A / TorchGT), P <= H.

Launch through the multi-node launcher:
    GT3D_SCRIPT=bench/bench_dist_attention.py srun ... bash scripts/multinode_train.sh --data <partition> --grid P 1 --strategy ring ...
Prints one ATTN line per run (rank 0): ms per layer forward / backward (max over ranks, mean over iterations 2+),
the same x `--layers`, and the peak memory of the operator above its inputs.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist.grid2d import Grid2D, all_gather_rows  # noqa: E402
from gt3d.dist.sparse_attention import AttentionBlock, DistAttentionPartial, combine_partials  # noqa: E402
from gt3d.data.loader import load_partition, read_meta  # noqa: E402
from gt3d.data.expander import block_expander_edges  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--grid", type=int, nargs="+", default=[4, 1], help="P_d P_o [P_i]  (P_i only with --strategy head)")
    ap.add_argument("--strategy", default="gather", choices=["gather", "gather_proj", "ring", "ring_kv", "ring_kv2", "head"])
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4, help="only scales the reported per-model number")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="triton", choices=["torch", "triton"])
    ap.add_argument("--comm_bf16", action="store_true")
    ap.add_argument("--rs_bf16", action="store_true", help="(gather) reduce-scatter the partials in bfloat16")
    ap.add_argument("--no_comm", action="store_true", help="replace every collective / hop by a local operation of the same shape (wrong result; measures compute only)")
    ap.add_argument("--sparse_comm", action="store_true", help="(gather / gather_proj) sparsity-aware collectives (gt3d/dist/sparse_comm.py): only the rows the tile references are gathered / reduce-scattered (all-to-all-v with static index sets)")
    ap.add_argument("--sparse_threshold", type=float, default=None, help="(--sparse_comm) forced share limit per collective (1.0 = every collective sparse); default: the group layout's a2a / all-gather rate x margin")
    ap.add_argument("--profile", action="store_true", help="torch.profiler on iteration 3: device kernel time by category (NCCL / attention / GEMM / other), max and mean over ranks")
    ap.add_argument("--coll_timers", action="store_true", help="CUDA-event timing of EVERY collective call (gathers, reduce-scatters, all-to-alls) in program order: per call the duration max / mean / min over the ranks and the compute gap before it (mean over the timed iterations)")
    args = ap.parse_args()
    if args.strategy in ("ring", "ring_kv", "ring_kv2"):
        assert args.grid[1] == 1, "ring attention needs a (P, 1) grid"
    p_i = args.grid[2] if len(args.grid) == 3 else 1
    assert args.strategy == "head" or p_i == 1, "a head axis (third grid dimension) needs --strategy head"
    grid_str = "x".join(map(str, args.grid))

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)

    meta = read_meta(args.data)
    n = meta["num_nodes"]
    if args.strategy == "head":
        from gt3d.dist.grid3d import Grid3D
        grid = Grid3D(args.grid[0], args.grid[1], p_i, n, n_pad=meta["n_pad"]).to(dev)
    else:
        grid = Grid2D(args.grid[0], args.grid[1], n, n_pad=meta["n_pad"]).to(dev)
    REC = []                                                         # (name, bytes, start event, end event) per collective call
    if args.coll_timers:
        import gt3d.dist.grid2d as _g2
        import gt3d.dist.grid3d as _g3

        def _label(group):
            if group is grid.row_group:
                return "row"
            if group is grid.col_group:
                return "col"
            if getattr(grid, "head_group", None) is group:
                return "head"
            return "?"

        def _wrap(fn, kind):
            def w(x, group, *a, **k):
                s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                out = fn(x, group, *a, **k)
                e_ev.record()
                REC.append((f"{kind}({_label(group)})", x.numel() * x.element_size(), s_ev, e_ev))
                return out
            return w
        _g2.raw_all_gather_rows = _wrap(_g2.raw_all_gather_rows, "AG")
        _g2.raw_reduce_scatter_rows = _wrap(_g2.raw_reduce_scatter_rows, "RS")
        _g3.raw_rows_to_heads = _wrap(_g3.raw_rows_to_heads, "A2A")
        _g3.raw_heads_to_rows = _wrap(_g3.raw_heads_to_rows, "A2A")
    part = load_partition(args.data, grid, dev, load_weights=False)
    gcn_blk = part["edge_index"]                                     # original edges incl. self loops (type 0)
    exp_blk = block_expander_edges(grid, n, args.exp_deg, seed=args.seed, device=dev)
    attn_blk = torch.cat([gcn_blk, exp_blk], dim=1)
    attn_et = torch.cat([torch.zeros(gcn_blk.shape[1], dtype=torch.long, device=dev),
                         torch.ones(exp_blk.shape[1], dtype=torch.long, device=dev)])
    del part, gcn_blk, exp_blk
    n_edges_local = int(attn_blk.shape[1])

    hdim, h = args.hidden, args.heads
    d = hdim // h
    hd = h * d
    chunk = grid.chunk
    cdt = torch.bfloat16 if args.comm_bf16 else None
    rs_dtype = torch.bfloat16 if args.rs_bf16 else None
    # inputs / parameters of one attention layer (random; timing only)
    x = torch.randn(chunk, hdim, device=dev, requires_grad=True)
    sc = 1.0 / hdim ** 0.5
    wq = torch.nn.Parameter(torch.randn(hdim, hdim, device=dev) * sc)
    wk = torch.nn.Parameter(torch.randn(hdim, hdim, device=dev) * sc)
    wv = torch.nn.Parameter(torch.randn(hdim, hdim, device=dev) * sc)
    gate = torch.nn.Parameter(torch.randn(2, h, d, device=dev))
    g_out = torch.randn(chunk, h, d, device=dev)
    params = [wq, wk, wv, gate]

    if args.strategy in ("gather", "gather_proj"):
        tile = AttentionBlock(grid, attn_blk, attn_et)
        sp_row = sp_col = None
        if args.sparse_comm:
            from gt3d.models.exphormer_dist import build_sparse_comm, _gather_rows
            sp, info = build_sparse_comm(grid, tile, None, args.sparse_threshold)
            if rank == 0:
                print(f"[sparse_comm] {info}", flush=True)
            if sp is not None:
                tile, sp_row, sp_col = sp["attn"], sp["row"], sp["col"]
        attn = DistAttentionPartial(grid, tile, h, d, backend=args.backend, score="gated", norm="clampexp")
        del attn_blk, attn_et

        g_col, g_row = dist.get_world_size(grid.col_group), dist.get_world_size(grid.row_group)
        n_row_blk = sp_row.n_need if sp_row is not None else g_row * chunk          # rows of the (compact) row block
        n_col_blk = sp_col.n_need if sp_col is not None else g_col * chunk

        def gather(t, group, plan):
            if plan is None:
                return all_gather_rows(t, group)
            from gt3d.dist.sparse_comm import sparse_all_gather
            return sparse_all_gather(t, plan)

        def local_block(t, n_rows):                                             # --no_comm stand-in of a gathered block
            reps = -(-n_rows // t.shape[0])
            return t.repeat(reps, 1)[:n_rows]

        def finish(acc, l, m):
            if args.no_comm:
                n_rows = acc.shape[0]
                packed = torch.cat([acc.reshape(n_rows, -1), l], dim=1)
                if rs_dtype is not None:
                    packed = packed.to(rs_dtype)
                if sp_row is not None:                                              # compact rows: local index_add stand-in
                    packed = packed.new_zeros(chunk, packed.shape[1]).index_add_(0, sp_row.send_idx, local_block(packed, sp_row.n_send)).to(acc.dtype)
                else:
                    packed = packed.view(g_row, n_rows // g_row, -1).sum(0).to(acc.dtype)   # local "reduce-scatter"
                acc_o = packed[:, : h * d].view(-1, h, d)
                l_o = packed[:, h * d:]
                return acc_o / (l_o + 1e-6).unsqueeze(-1)
            o, _ = combine_partials(acc, l, m, grid, "clampexp", 1e-6, extra=None, rs_dtype=rs_dtype, plan=sp_row)
            return o

        def forward_proj(x):                                                # owner projects, projections gathered
            q_loc = x @ wq.t()
            kv_loc = torch.cat([x @ wk.t(), x @ wv.t()], dim=1)              # [chunk, 2 hd]
            if cdt is not None:
                q_loc, kv_loc = q_loc.to(cdt), kv_loc.to(cdt)
            if args.no_comm:
                q_row, kv_col = local_block(q_loc, n_row_blk), local_block(kv_loc, n_col_blk)
            else:
                q_row = gather(q_loc, grid.row_group, sp_row)               # Q side: row block, width hd
                kv_col = gather(kv_loc, grid.col_group, sp_col)             # K/V side: column block, width 2 hd
            q = q_row.to(x.dtype).view(-1, h, d)
            k = kv_col[:, :hd].to(x.dtype).contiguous().view(-1, h, d)
            v = kv_col[:, hd:].to(x.dtype).contiguous().view(-1, h, d)
            return finish(*attn(q, k, v, gate))

        def forward_x(x):
            xc = x.to(cdt) if cdt is not None else x
            if args.no_comm:                                                # local stand-ins, same shapes
                x_col = local_block(xc, n_col_blk).to(x.dtype)
                x_row = local_block(xc, n_row_blk).to(x.dtype)
            else:
                x_col = gather(xc, grid.col_group, sp_col).to(x.dtype)     # K/V side: column block
                x_row = gather(xc, grid.row_group, sp_row).to(x.dtype)     # Q side: row block
            q = (x_row @ wq.t()).view(-1, h, d)
            k = (x_col @ wk.t()).view(-1, h, d)
            v = (x_col @ wv.t()).view(-1, h, d)
            return finish(*attn(q, k, v, gate))

        forward = forward_proj if args.strategy == "gather_proj" else forward_x
    elif args.strategy == "head":
        from gt3d.dist.sparse_attention import DistHeadSparseAttention
        attn = DistHeadSparseAttention(grid, AttentionBlock(grid, attn_blk, attn_et), h, d, backend=args.backend,
                                       score="gated", norm="clampexp", comm_dtype=cdt, rs_dtype=rs_dtype,
                                       no_comm=args.no_comm)
        del attn_blk, attn_et

        def forward(x):
            q = (x @ wq.t()).view(-1, h, d)                               # owner projects its chunk, all heads
            k = (x @ wk.t()).view(-1, h, d)
            v = (x @ wv.t()).view(-1, h, d)
            return attn(q, k, v, gate)
    elif args.strategy in ("ring_kv", "ring_kv2"):
        from gt3d.dist.ring_kv import DistRingAttentionKV
        attn = DistRingAttentionKV.from_block(grid, attn_blk, attn_et, h, d, backend=args.backend, score="gated",
                                              norm="clampexp", comm_dtype=cdt, no_comm=args.no_comm,
                                              bidir=args.strategy == "ring_kv2")
        del attn_blk, attn_et

        def forward(x):
            q = (x @ wq.t()).view(-1, h, d)                               # owner projects its chunk once
            k = (x @ wk.t()).view(-1, h, d)
            v = (x @ wv.t()).view(-1, h, d)
            return attn(q, k, v, gate)
    else:
        from gt3d.dist.ring import DistExphormerLayerRing, _RingAttnGCN, split_block_by_source_chunk
        parts = split_block_by_source_chunk(attn_blk, grid.p_rows, chunk, attn_et, None)
        attn_blocks = [AttentionBlock(grid, ei, et) for ei, et, _ in parts]
        del attn_blk, attn_et, parts
        layer = DistExphormerLayerRing(grid, [None] * grid.p_rows, attn_blocks, hdim, h, 0.0, 1e-5, 0.1, args.backend,
                                       comm_dtype=cdt)
        layer.with_gcn = False
        layer.no_comm = args.no_comm

        def forward(x):
            packed = _RingAttnGCN.apply(x, wq, wk, wv, None, gate, layer)
            acc = packed[:, :hd].view(-1, h, d)
            l = packed[:, hd:hd + h]
            return acc / (l + 1e-6).unsqueeze(-1)

    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    t_fwd, t_bwd = [], []
    import contextlib
    prof_tot = None
    per_iter = []                                                    # coll_timers: records of every timed iteration
    for it in range(args.iters):
        if it == 2:
            torch.cuda.reset_peak_memory_stats()
        REC.clear()
        prof_ctx = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) \
            if (args.profile and it == 3) else contextlib.nullcontext()
        torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        with prof_ctx as prof:
            o = forward(x)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            (o * g_out).sum().backward()
            torch.cuda.synchronize(); t2 = time.perf_counter()
        if args.profile and it == 3:
            from gt3d.train.train_dist import _profile_totals
            prof_tot = _profile_totals(prof)
        x.grad = None
        for p in params:
            p.grad = None
        if it >= 2:
            t_fwd.append(t1 - t0); t_bwd.append(t2 - t1)
            if args.coll_timers:
                torch.cuda.synchronize()
                per_iter.append([(n, b, s_.elapsed_time(e_), (per_iter_prev_end.elapsed_time(s_) if i > 0 else 0.0))
                                 for i, (n, b, s_, e_) in enumerate(REC)
                                 for per_iter_prev_end in [REC[i - 1][3] if i > 0 else s_]])
    if args.coll_timers and per_iter:
        n_calls = min(len(r) for r in per_iter)
        mine = [(per_iter[0][i][0], per_iter[0][i][1], sum(r[i][2] for r in per_iter) / len(per_iter), sum(r[i][3] for r in per_iter) / len(per_iter))
                for i in range(n_calls)]
        allr = [None] * world
        dist.all_gather_object(allr, mine)
        if rank == 0:
            print(f"COLL strategy={args.strategy} grid={grid_str}: per collective call in program order (forward then backward), "
                  f"mean over {len(per_iter)} timed iterations; duration ms max / mean / min over {world} ranks; gap = compute before the call (mean / max over ranks)", flush=True)
            tot_max = tot_mean = 0.0
            for i in range(n_calls):
                name, nb = allr[0][i][0], allr[0][i][1]
                d = [r[i][2] for r in allr]; g = [r[i][3] for r in allr]
                tot_max += max(d); tot_mean += sum(d) / world
                print(f"COLL {i:2d} {name:10s} {nb/2**20:7.1f} MB  dur {max(d):7.2f} / {sum(d)/world:7.2f} / {min(d):7.2f} ms   gap {sum(g)/world:6.2f} / {max(g):6.2f} ms", flush=True)
            print(f"COLL total collective time: max-over-ranks sum {tot_max:.1f} ms, mean-over-ranks sum {tot_mean:.1f} ms", flush=True)
    peak_extra = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    stats = torch.tensor([sum(t_fwd) / len(t_fwd), sum(t_bwd) / len(t_bwd), peak_extra, peak], device=dev, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    e_tot = torch.tensor([n_edges_local], device=dev); dist.all_reduce(e_tot); e_tot = e_tot // p_i   # each block is held by P_i ranks
    if prof_tot is not None:
        keys = ["total", "nccl", "attn_fwd", "attn_bwd", "gemm", "other"]
        pt = torch.tensor([prof_tot[k] / 1e3 for k in keys], device=dev, dtype=torch.float64)
        p_max = pt.clone(); dist.all_reduce(p_max, op=dist.ReduceOp.MAX)
        p_mean = pt.clone(); dist.all_reduce(p_mean); p_mean /= world
        if rank == 0:
            print("PROF " + " ".join(f"{k}={p_max[i].item():.1f}/{p_mean[i].item():.1f}" for i, k in enumerate(keys))
                  + f"  (device kernel ms for one layer fwd+bwd, max/mean over ranks; strategy={args.strategy} grid={grid_str} data={os.path.basename(args.data)} hidden={hdim})", flush=True)
    if rank == 0:
        fwd_ms, bwd_ms = stats[0].item() * 1e3, stats[1].item() * 1e3
        print(f"ATTN strategy={args.strategy}{'-nocomm' if args.no_comm else ''}{'-sparse' if args.sparse_comm else ''} grid={grid_str} gpus={world} data={os.path.basename(args.data)} "
              f"hidden={hdim} heads={h} bf16={int(args.comm_bf16)} rs_bf16={int(args.rs_bf16)} E_attn={int(e_tot.item())} "
              f"fwd_ms={fwd_ms:.1f} bwd_ms={bwd_ms:.1f} layer_ms={fwd_ms + bwd_ms:.1f} x{args.layers}layers_ms={(fwd_ms + bwd_ms) * args.layers:.1f} "
              f"peak_extra_gb={stats[2].item():.2f} peak_gb={stats[3].item():.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
