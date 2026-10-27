"""Multi-GPU full-graph Exphormer training on ogbn-arxiv with the 2D grid (official arxiv.yaml recipe).

    torchrun --nproc_per_node 4 gt3d/train/train_dist.py --grid 2 2 --backend triton --epochs 600

Every rank loads the graph (small enough here), builds its local blocks, and owns a node chunk for
all dense work.  Parameters are replicated; gradients are summed with one flat all-reduce per step.
"""
import argparse
import re
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gt3d.models.exphormer_ref import ExphormerRef  # noqa: E402
from gt3d.models.exphormer_dist import DistExphormer, gcn_edges_with_self_loops  # noqa: E402
from gt3d.dist.grid2d import Grid2D  # noqa: E402
from gt3d.dist.layers import dist_cross_entropy, dist_accuracy, allreduce_grads, set_step  # noqa: E402
from axonn import axonn as ax  # noqa: E402
from plexus.utils.general import print_axonn_timer_data  # noqa: E402
from gt3d.data.expander import random_regular_expander_edges  # noqa: E402
from gt3d.train.train_single_gpu import load_arxiv, cosine_with_warmup  # noqa: E402


def _profile_totals(prof):
    """Aggregate CUDA kernel time (us) by category for one training step (kernels only)."""
    cats = {"attn_fwd": ("_attn_fwd_kernel",), "attn_bwd": ("_attn_bwd_kernel", "_attn_bwd_kv_kernel"),
            "spmm": ("cusparse", "spmm", "csrmm", "Csr"), "nccl": ("ncclKernel", "ncclDevKernel"),
            "gemm": ("gemm", "cutlass", "sgemm", "Gemm"), "segment": ("segment", "scatter", "index_", "gather", "Index")}
    tot = {k: 0.0 for k in cats}; tot["other"] = 0.0; tot["total"] = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        k = ev.key
        # autograd Function wrappers (CamelCase after an underscore, e.g. _AttnPartial, _RowsToHeads) carry the device time of
        # the kernels they launch: skip them, otherwise that time is counted twice (Triton kernels are lower-case)
        if t <= 0 or re.match(r"^_[A-Z]", k) or k.startswith(("nccl:", "aten::", "autograd::", "torch", "Optimizer", "_Dist", "_Masked", "_SpMM", "_AllGather",
                                   "_ReduceScatter", "_AllReduce", "cuda", "record_param", "ProfilerStep", "enumerate", "<built-in",
                                   "nn.Module", "gt3d", "Checkpoint", "CheckpointFunction")):
            continue                                        # device kernels only (skip CPU op wrappers)
        tot["total"] += t
        for k, pats in cats.items():
            if any(pt in ev.key for pt in pats):
                tot[k] += t; break
        else:
            tot["other"] += t
    return tot


def _print_profile(prof, rank, world):
    tot = _profile_totals(prof)
    gathered = [None] * world
    dist.all_gather_object(gathered, tot)
    if rank == 0:
        keys = ["total", "attn_fwd", "attn_bwd", "nccl", "gemm", "spmm", "segment", "other"]
        print("[prof] per-rank device kernel time (ms) for one step: " + "  ".join(f"{k:>9s}" for k in keys), flush=True)
        for r, t in enumerate(gathered):
            print(f"[prof] rank {r}: " + "  ".join(f"{t[k]/1e3:9.1f}" for k in keys), flush=True)
        mx = max(t["total"] - t["nccl"] for t in gathered); mn = min(t["total"] - t["nccl"] for t in gathered)
        print(f"[prof] compute (total - nccl) max/min over ranks = {mx/max(mn,1):.2f}x  (load imbalance)", flush=True)
        top = sorted(((getattr(ev, "self_device_time_total", 0.0) or 0.0, ev.key) for ev in prof.key_averages()
                      if not ev.key.startswith(("nccl:", "aten::", "autograd::", "torch", "_Dist", "_Masked", "_SpMM", "_All", "_Reduce", "cuda", "record_param"))),
                     reverse=True)[:14]
        for t, k in top:
            print(f"[prof]     {t/1e3:8.1f} ms  {k[:110]}", flush=True)


class _saved_census:
    """Context manager: records every tensor autograd saves for backward during the forward (saved_tensors_hooks),
    de-duplicates by storage and classifies the bytes by first dimension (owned chunk O, row block R = P_o chunk,
    column block C = P_d chunk, local attention / GCN edges) and dtype.  Prints the totals and the multipliers in
    units of U_x = rows_x * hidden * 4 B: the counted (not fitted) retained-activation term of the memory model."""

    def __init__(self, grid, model, rank):
        self.grid, self.model, self.rank = grid, model, rank
        self.seen, self.items = set(), []

    def _pack(self, t):
        st = t.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        if key not in self.seen:
            self.seen.add(key)
            self.items.append((tuple(t.shape), t.dtype, st.nbytes()))
        return t

    def __enter__(self):
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)
        g = self.grid
        c_lo, c_hi = g.col_block_range()
        classes = {"owned O": g.chunk, "row block R": g.n_row_block, "col block C": c_hi - c_lo}   # Grid3D: P_i chunks per rank of the group
        edges = {}
        for L in getattr(self.model, "layers", []):
            for name in ("attn", "gcn_block"):
                b = getattr(L, name, None)
                for attr in ("num_edges", "n_edges"):
                    if b is not None and hasattr(b, attr):
                        edges[f"edges ({name})"] = getattr(b, attr)
                if b is not None and hasattr(b, "col"):
                    edges[f"edges ({name})"] = int(b.col.numel())
        classes.update(edges)
        by = {}
        for shape, dtype, nbytes in self.items:
            n0 = shape[0] if shape else 0
            cls = next((k for k, v in classes.items() if v == n0), "other")
            key = (cls, str(dtype).replace("torch.", ""))
            cnt, tot = by.get(key, (0, 0))
            by[key] = (cnt + 1, tot + nbytes)
        if self.rank == 0:
            hidden = getattr(self.model, "dim_hidden", None) or getattr(self.model.layers[0], "dim_h", 0)
            print(f"[census] saved tensors (unique storages) during the forward: {len(self.items)}, "
                  f"{sum(n for *_, n in self.items)/2**30:.3f} GB; classes {classes}", flush=True)
            for (cls, dt), (cnt, tot) in sorted(by.items(), key=lambda kv: -kv[1][1]):
                unit = classes.get(cls)
                mult = f"  = {tot / (unit * hidden * 4):6.2f} x U({cls.split()[0]})" if unit and hidden and "edges" not in cls else ""
                print(f"[census]   {cls:18s} {dt:8s}: {cnt:4d} tensors {tot/2**30:7.3f} GB{mult}", flush=True)
            others = [(sh, str(dt).replace('torch.', ''), nb) for sh, dt, nb in self.items
                      if (sh[0] if sh else 0) not in classes.values()]
            others.sort(key=lambda x: -x[2])
            print("[census]   largest 'other' tensors: " + ", ".join(f"{sh}/{dt}/{nb/2**20:.1f}MB" for sh, dt, nb in others[:8]), flush=True)
        return False


class _coll_stats:
    """--coll_stats: CUDA-event timing of EVERY synchronous collective call (grid2d row all-gathers / reduce-scatters, grid3d
    all-to-alls) in program order.  Every rank records (site, bytes, ms) per call; the calls are in the same order on every
    rank, so per call the max / mean / MIN over the ranks are well defined: the minimum is the time of the last-arriving
    rank, i.e. the collective itself, the gap to the mean / max is the arrival skew the collective absorbs.  Printed per
    call site (kind, group, forward / backward) as sums over the calls of an epoch, mean over the timed epochs."""

    def __init__(self, rank, world):
        import gt3d.dist.grid2d as g2
        import gt3d.dist.grid3d as g3
        self.rank, self.world, self.rec, self.epochs = rank, world, [], []
        self.in_backward = False

        def wrap(mod, name, kind):
            fn = getattr(mod, name)

            def w(x, group, *a, **k):
                if k.get("async_op") or (len(a) > 0 and a[0] is True and name == "raw_all_gather_rows"):
                    return fn(x, group, *a, **k)
                s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                out = fn(x, group, *a, **k)
                e_ev.record()
                site = f"{'bwd' if self.in_backward else 'fwd'} {kind}({g2.GROUP_NAMES.get(id(group), '?')})"
                self.rec.append((site, x.numel() * x.element_size(), s_ev, e_ev))
                return out
            setattr(mod, name, w)
        wrap(g2, "raw_all_gather_rows", "AG")
        wrap(g2, "raw_reduce_scatter_rows", "RS")
        wrap(g3, "raw_rows_to_heads", "A2A")
        wrap(g3, "raw_heads_to_rows", "A2A")

    def end_epoch(self, keep):
        torch.cuda.synchronize()
        if keep:
            self.epochs.append([(site, nb, s.elapsed_time(e)) for site, nb, s, e in self.rec])
        self.rec = []

    def report(self):
        if not self.epochs:
            return
        n_calls = min(len(ep) for ep in self.epochs)
        mine = [(self.epochs[0][i][0], self.epochs[0][i][1], sum(ep[i][2] for ep in self.epochs) / len(self.epochs)) for i in range(n_calls)]
        allr = [None] * self.world
        dist.all_gather_object(allr, mine)
        if self.rank != 0:
            return
        sites = {}
        for i in range(n_calls):
            site, nb = allr[0][i][0], allr[0][i][1]
            d = [r[i][2] for r in allr]
            agg = sites.setdefault(site, [0, 0.0, 0.0, 0.0, 0.0])
            agg[0] += 1; agg[1] += nb; agg[2] += max(d); agg[3] += sum(d) / self.world; agg[4] += min(d)
        print(f"[coll] per call site, sums over the calls of one epoch (mean over {len(self.epochs)} timed epochs); ms max / mean / min over "
              f"{self.world} ranks; GB/s = bytes moved by this rank / min time", flush=True)
        for site, (n, nb, mx, mean, mn) in sites.items():
            print(f"[coll] {site:14s} calls {n:2d}  {nb/2**20:8.1f} MB/epoch  max {mx:7.2f}  mean {mean:7.2f}  min {mn:7.2f} ms   "
                  f"skew(mean-min) {mean-mn:6.2f}   {nb/max(mn,1e-6)/1e6:6.1f} GB/s", flush=True)


def graph_name(args):
    """Key of grid_select.GRAPH_LOCALITY for the data set: --graph_name, else the partition directory's name before
    '_p' (products_p16b -> products), else 'arxiv'."""
    if args.graph_name:
        return args.graph_name
    if args.data is None:
        return "arxiv"
    base = os.path.basename(os.path.normpath(args.data))
    return base.split("_p")[0] if "_p" in base else base


def auto_grid(world, num_nodes, num_edges, args, rank, f_in):
    """Pick (P_d, P_o) for this world size: fastest grid under the white-box performance model (bytes / FLOPs / rows x
    measured hardware constants, gt3d/dist/grid_select.py) among those the counted memory model says fit in 40 GB
    (streamed layer: memory O(N/P_d + N/(P_o n_sub)); fused: O(N/P_o + N/P_d)).  Row-major layout, fused layer, bf16
    gathers and TF32 matmuls assumed."""
    from gt3d.dist.grid_select import select_grid_mem, attention_edges
    e_attn = attention_edges(num_nodes, num_edges, args.exp_deg)
    e_gcn = num_edges + num_nodes
    best, ranked = select_grid_mem(world, num_nodes, e_attn, e_gcn, args.hidden, f_in, args.layers,
                                   n_sub=max(args.stream_sub, 1), streamed=args.stream_sub > 1, rs_bf16=args.rs_bf16,
                                   heads=args.heads, tf32=not args.no_tf32_matmul, graph=graph_name(args))
    if best is None:
        best = ranked[0]                                   # nothing fits by the model: take the fastest and let it try
        note = " (WARNING: no grid predicted to fit in 40 GB)"
    else:
        note = ""
    pd, po, ms, mem = best
    if rank == 0:
        alts = ", ".join(f"{a}x{b} {t:.0f} ms/{m:.0f} GB" for a, b, t, m in ranked[:5] if (a, b) != (pd, po))
        print(f"[grid_auto] world={world} N={num_nodes} E_attn={e_attn} hidden={args.hidden} stream_sub={args.stream_sub}: "
              f"grid {pd}x{po} (predicted {ms:.0f} ms/epoch, {mem:.1f} GB peak){note}; others: {alts}", flush=True)
        if args.grid_layout != "row_major":
            print("[grid_auto] warning: the model assumes the row_major layout", flush=True)
    return [pd, po]


def nccl_tune(args, grid_dims, num_nodes, world, rank, gpn=4, num_edges=None):
    """--nccl_tune auto: raise NCCL's channel count (NCCL_MIN_NCHANNELS=16, NCCL_ALGO=Ring) when the column group leaves
    the node and its per-rank reduce-scatter payload is large.  In the training process the cross-node reduce-scatter runs
    at 70-80 % of its micro-benchmark rate (CPU-side progress of the NCCL proxy under the training threads, RESULTS
    2026-10-18); 16 channels with the ring hide most of it: products at 64 GPUs 16x4 / 8x8 / 32x2 −4 % per epoch, while
    the same setting costs small-message runs 20-40 % (Reddit at 64 GPUs: 0.9 MB chunks), hence the payload threshold.
    Must run BEFORE the first NCCL collective (the communicators read the environment when they are created).
    With --sparse_comm the tuning is skipped when the column group is predicted to go sparse (the analytic share rule of
    grid_select.sparse_share, i.e. the decision build_sparse_comm will take): NCCL_MIN_NCHANNELS=16 + NCCL_ALGO=Ring slow
    the all-to-all-v's send / recv path 1.5x while helping the ring collectives ~14 % (RESULTS 2026-10-23, 32-GPU
    diagnosis: 8x4 column-sparse 242.8 ms tuned vs 218.4 untuned); --nccl_tune on still forces it."""
    from gt3d.dist.grid_select import placement
    if args.nccl_tune == "off":
        return
    pd, po = grid_dims[0], grid_dims[1]
    pi = grid_dims[2] if len(grid_dims) == 3 else 1
    if args.nccl_tune == "auto" and getattr(args, "sparse_comm", False):
        from gt3d.dist.grid_select import Graph, attention_edges, sparse_share
        from gt3d.dist.sparse_comm import A2A_OVER_AG, MARGIN
        k_col, m_col = placement(pd, po, pi, gpn)["col"]
        # The plan decides on the MEASURED share, which the Poisson estimate overstates on skewed graphs (products 16x4:
        # analytic 0.975 vs measured 0.798), so on the layouts where a column plan is plausible at all (one rank per node,
        # a2a / AG 0.89) the tuning is skipped outright; whole-node column groups (ratio 0.17 / 0.40, never sparse below
        # P_d >= 256) keep it unless even the analytic share passes the rule.
        thr = args.sparse_threshold if args.sparse_threshold is not None else \
            A2A_OVER_AG.get((min(k_col, 4), m_col > 1), A2A_OVER_AG[(4, m_col > 1)]) * MARGIN
        if num_edges is None:
            share = float("nan")
            col_sparse = m_col > 1                                                   # unknown edge count: be conservative
        else:
            share = sparse_share(Graph(num_nodes, attention_edges(num_nodes, num_edges, args.exp_deg), num_edges, 1), pd, args.exp_deg)
            col_sparse = share <= thr or (k_col == 1 and m_col > 1 and thr > 0)
        if col_sparse:
            if rank == 0:
                print(f"[nccl_tune] --sparse_comm: column group ({k_col} rank(s) per node x {m_col} nodes, analytic share {share:.3f}, "
                      f"limit {thr:.2f}) may go SPARSE -> NCCL defaults (the channel tuning slows the all-to-all-v 1.5x)", flush=True)
            return
    b = 2 if args.comm_bf16 else 4
    o = num_nodes / world
    if pi == 1:
        payload = (num_nodes / po - o) * args.hidden * b                       # dx of the column block, per layer
    else:
        payload = (num_nodes / po - pi * o) * (args.hidden / pi) * b * (2 if args.no_gcn else 3)   # d[K | V | GCN msg] slices, one call
    k, m = placement(pd, po, pi, gpn)["col"]
    want = args.nccl_tune == "on" or (m >= 2 and payload >= 32 * 2 ** 20)
    if want:
        os.environ.setdefault("NCCL_MIN_NCHANNELS", "16")
        os.environ.setdefault("NCCL_ALGO", "Ring")
    if rank == 0:
        print(f"[nccl_tune] column group layout {k}x{m}, reduce-scatter payload {payload/2**20:.0f} MB per rank -> "
              f"{'NCCL_MIN_NCHANNELS=' + os.environ.get('NCCL_MIN_NCHANNELS', '?') + ' NCCL_ALGO=' + os.environ.get('NCCL_ALGO', '?') if want else 'NCCL defaults'}", flush=True)


def auto_head_grid(world, num_nodes, num_edges, args, rank, f_in):
    """Pick the layout for this world size with the white-box model: the fastest head grid (P_d, P_o, P_i)
    (gt3d/dist/grid_select.select_grid_head) against the fastest fused 2D grid of the same model (with or without the GCN
    branch, as configured), among the shapes the counted memory models say fit in 40 GB.  Returns [P_d, P_o, P_i] for the
    head layer or [P_d, P_o] for the fused layer (small per-rank problems, e.g. Reddit at 64 GPUs, are collective-latency
    bound and the fused layer's 3 collectives per layer and direction beat the head layer's 5)."""
    from gt3d.dist.grid_select import (select_grid_head, attention_edges, candidate_grids, admissible, predict_fused_ms,
                                       peak_memory_bytes, Graph, Model, GB)
    e_attn = attention_edges(num_nodes, num_edges, args.exp_deg)
    e_gcn = num_edges + num_nodes
    tf32 = not args.no_tf32_matmul
    with_gcn = not args.no_gcn
    gname = graph_name(args)
    best, ranked = select_grid_head(world, num_nodes, e_attn, args.hidden, f_in, args.layers, heads=args.heads,
                                    tf32=tf32, kernel_bf16=args.kernel_bf16, with_gcn=with_gcn, e_gcn=e_gcn, graph=gname)
    g, m = Graph(num_nodes, e_attn, e_gcn, f_in, name=gname), Model(args.hidden, args.heads, args.layers, tf32, with_gcn=with_gcn)
    fused = sorted(((pd, po, predict_fused_ms(pd, po, g, m),
                     peak_memory_bytes(pd, po, num_nodes, e_attn, e_gcn if with_gcn else 0, args.hidden, f_in, args.layers, heads=args.heads,
                                       kernel_bf16=args.kernel_bf16) / GB)
                    for pd, po in candidate_grids(world) if admissible(pd, po)), key=lambda r: r[2])
    fused_fit = [r for r in fused if r[3] <= 40.0 * 0.92]
    note = ""
    if best is None:
        best, note = ranked[0], " (WARNING: no head grid predicted to fit in 40 GB)"
    pd, po, pi, ms, mem = best
    f_best = fused_fit[0] if fused_fit else fused[0]
    use_fused = f_best[2] < ms and (fused_fit or best is ranked[0])
    if rank == 0:
        alts = ", ".join(f"({a},{b},{c}) {t:.0f} ms/{m_:.1f} GB" for a, b, c, t, m_ in ranked[:6] if (a, b, c) != (pd, po, pi))
        falts = ", ".join(f"{a}x{b} {t:.0f} ms/{m_:.1f} GB" for a, b, t, m_ in fused[:3])
        print(f"[head_auto] world={world} N={num_nodes} E_attn={e_attn} hidden={args.hidden} heads={args.heads} gcn={int(with_gcn)}: best head grid ({pd},{po},{pi}) "
              f"(predicted {ms:.0f} ms/epoch, {mem:.1f} GB peak){note}; other head grids: {alts}; fused 2D: {falts} "
              f"-> {'FUSED ' + str(f_best[0]) + 'x' + str(f_best[1]) if use_fused else 'HEAD (' + str(pd) + ',' + str(po) + ',' + str(pi) + ')'}", flush=True)
    return [f_best[0], f_best[1]] if use_fused else [pd, po, pi]


def _sparse_tag(args, model):
    """STEADY field: 0 (off), or r<0|1>c<0|1> = which collectives went sparse (row group / column group)."""
    if not args.sparse_comm:
        return "0"
    inf = getattr(model, "sparse_info", None) or {}
    return f"r{int(bool(inf.get('use_row')))}c{int(bool(inf.get('use_col')))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[2, 2],
                    help="P_d P_o (2D block grid) or P_d P_o P_i (Grid3D: 2D blocks x Ulysses head axis, P_i | heads; attention-only model, needs --no_gcn)")
    ap.add_argument("--backend", default="triton", choices=["torch", "triton"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--exp_deg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--data", default=None, help="gt3d partition directory (default: ogbn-arxiv in memory)")
    ap.add_argument("--checkpoint", action="store_true", help="activation checkpointing per layer")
    ap.add_argument("--profile", action="store_true", help="torch.profiler on epoch 2 (rank 0 prints a kernel-time breakdown)")
    ap.add_argument("--comm_bf16", action="store_true", help="gather Q/K/V (and o, dO in backward) in bfloat16")
    ap.add_argument("--fused", action="store_true", help="fused-communication layer (gather x once, one reduce-scatter)")
    ap.add_argument("--compile", action="store_true", help="torch.compile every layer (dense parts; collectives/kernels stay opaque)")
    ap.add_argument("--rs_bf16", action="store_true", help="(fused layer) reduce-scatter the partial sums in bfloat16")
    ap.add_argument("--pipeline_sub", type=int, default=1, help="(fused layer) gather the column block in this many sub-blocks, overlapping compute")
    ap.add_argument("--stream_sub", type=int, default=1, help="(fused layer) stream the column block in this many sub-blocks, recomputed in backward (memory O(N/P_d + N/(P_o n_sub))); with --checkpoint only the owned-row tail is checkpointed")
    ap.add_argument("--ring", action="store_true", help="ring attention on a (P, 1) grid: input chunks rotate around the ring (P2P), K/V projected per hop and recomputed in backward; memory O(N/P) (gt3d/dist/ring.py)")
    ap.add_argument("--grid_layout", default="row_major", choices=["row_major", "col_major"], help="rank -> (row, col) mapping")
    ap.add_argument("--norm", default="bn", choices=["bn", "ln", "rms"], help="bn: Exphormer BatchNorm with global statistics; ln: LayerNorm; rms: Plexus ScatteredRMSNorm (both without collectives)")
    ap.add_argument("--timers", action="store_true", help="AxoNN CUDA-event timers per region; tree printed after the timing epochs (rank 0)")
    ap.add_argument("--graph_name", default=None, help="key of the per-graph kernel locality coefficients (grid_select.GRAPH_LOCALITY) used by --grid_auto / --head_auto; default: the partition directory's name before '_p'")
    ap.add_argument("--nccl_tune", default="auto", choices=["auto", "on", "off"], help="auto (default): NCCL_MIN_NCHANNELS=16 + NCCL_ALGO=Ring when the column group spans nodes and its per-rank reduce-scatter payload is >= 32 MB (the in-situ reduce-scatter deficit, RESULTS 2026-10-18); on / off force it")
    ap.add_argument("--coll_stats", action="store_true", help="CUDA-event timing of every synchronous collective call: per call site the max / mean / min over ranks (min = the collective itself, mean - min = arrival skew), epochs 2+ (rank 0)")
    ap.add_argument("--saved_census", action="store_true", help="epoch 2: count the tensors autograd saves for backward (unique storages) by row class (owned / row block / column block / edges) and dtype; the counted retained-activation term of the memory model (rank 0)")
    ap.add_argument("--auto_ckpt", action="store_true", help="enable activation checkpointing when the memory model predicts > 34 GB without it")
    ap.add_argument("--grid_auto", action="store_true", help="choose the grid for the world size with the performance model (gt3d/dist/grid_select.py); overrides --grid")
    ap.add_argument("--head_auto", action="store_true", help="choose the layout for the world size with the white-box model over BOTH families: the head grid (P_d, P_o, P_i) and the fused 2D grid, for the model as configured (with / without the GCN branch); overrides --grid")
    ap.add_argument("--timing_epochs", type=int, default=0, help="if > 0: run this many epochs, no eval except the last")
    ap.add_argument("--no_gcn", action="store_true", help="attention-only Exphormer (no parallel GCN branch, GraphGPS local_gnn_type=None); plain / fused layers only")
    ap.add_argument("--stream_overlap", action="store_true", help="(streamed layer) prefetch the next column sub-block and reduce-scatter dx asynchronously (communication / compute overlap); the default since 2026-10-15 (-21..-28 %% on papers100M at 64 GPUs), kept for old scripts")
    ap.add_argument("--no_stream_overlap", action="store_true", help="(streamed layer) synchronous sub-block gathers and dx reduce-scatters (the pre-2026-10-15 behaviour)")
    ap.add_argument("--tf32_matmul", action="store_true", help="TF32 tensor cores for the fp32 dense matmuls (projections, FFN); the default since 2026-10-16 (accuracy unchanged on arxiv / Reddit / products, -18 %% at hidden 128-256), kept for old scripts")
    ap.add_argument("--no_tf32_matmul", action="store_true", help="IEEE fp32 dense matmuls (cuBLAS default; the pre-2026-10-16 behaviour)")
    ap.add_argument("--kernel_bf16", action="store_true", help="store Q / K / V (and the dacc gradient) in bfloat16 for the attention kernels (fp32 accumulation inside): half the per-edge kernel bytes and the saved K / V per layer; fused and streamed layers")
    ap.add_argument("--no_stream_compact", action="store_true", help="(streamed layer, triton) full-row-block attention partials + add instead of the compact-accumulate kernels (only the rows with edges in the sub-block are tiled, partials added in place)")
    ap.add_argument("--gather_proj", action="store_true", help="(fused layer) owner projects Q/K/V/GCN message once and the projections are gathered instead of x (no redundant projection of the gathered blocks; 3x the column-group bytes)")
    ap.add_argument("--model", default="exphormer", choices=["exphormer", "unimp", "graphormer"],
                    help="exphormer (GCN + expander attention, default) | unimp (TransformerConv + gated residual, 1-hop mask, softmax) | graphormer (node-level Graphormer: 1-hop + global token, relation bias, pre-LN; gt3d/models/gt_variants.py)")
    ap.add_argument("--label_input", action="store_true", help="(unimp) masked label prediction: the labels of a random --label_rate share of the training nodes are fed as input features, the loss is taken on the rest; all training labels at eval")
    ap.add_argument("--label_rate", type=float, default=0.5)
    ap.add_argument("--max_degree", type=int, default=64, help="(graphormer) in-degree embedding bins (centrality encoding)")
    ap.add_argument("--ffn_mult", type=int, default=2, help="(graphormer) FFN width / hidden")
    ap.add_argument("--no_attn", action="store_true", help="GCN-only model (no attention branch, GraphGPS global_model_type=None); plain layer only (no --fused)")
    ap.add_argument("--sparse_comm", action="store_true", help="sparsity-aware communication (gt3d/dist/sparse_comm.py; fused / streamed / head layers): every rank receives only the rows of the row / column block its tile references (static index sets, all-to-all-v instead of all-gather, the transpose for the reduce-scatters); a collective goes sparse only when its row share is below the group layout's a2a / all-gather rate x margin")
    ap.add_argument("--sparse_threshold", type=float, default=None, help="(--sparse_comm) force the share limit per collective: 1.0 = always sparse, 0 = never (default: the measured rate ratio of the group's node layout x GT3D_SPARSE_MARGIN)")
    ap.add_argument("--sparse_threshold_row", type=float, default=None, help="(--sparse_comm) share limit of the ROW group only (overrides --sparse_threshold for it; e.g. 0 with --sparse_threshold 1.0 = column plan only)")
    args = ap.parse_args()
    args.stream_overlap = not args.no_stream_overlap            # overlap is the default for the streamed layer
    args.tf32_matmul = not args.no_tf32_matmul
    if args.tf32_matmul:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.timing_epochs > 0:
        args.epochs = args.timing_epochs
        args.eval_every = args.timing_epochs
    if args.ring:
        assert args.grid[1] == 1 and not args.grid_auto, "--ring needs --grid P 1"
    if args.head_auto:
        args.grid = [1, 1, 1]                                      # placeholder: chosen once N / E are known
    assert len(args.grid) in (2, 3), args.grid
    head_grid = len(args.grid) == 3
    if head_grid:
        assert not (args.fused or args.ring or args.gather_proj or max(args.pipeline_sub, args.stream_sub) > 1 or args.grid_auto), \
            "the head grid has its own layer: no --fused / --ring / --gather_proj / --pipeline_sub / --stream_sub / --grid_auto"
        assert args.heads % args.grid[2] == 0, (args.heads, args.grid)
        assert args.model == "exphormer", "the head grid is implemented for the Exphormer model"

    def make_grid(n, n_pad=None):
        if head_grid:
            from gt3d.dist.grid3d import Grid3D
            return Grid3D(args.grid[0], args.grid[1], args.grid[2], n, n_pad=n_pad, timers=args.timers).to(dev)
        return Grid2D(args.grid[0], args.grid[1], n, n_pad=n_pad, layout=args.grid_layout, timers=args.timers).to(dev)

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)                       # identical parameter init on every rank

    gcn_w = None
    if args.data is None:
        n, x, y, ei, split, num_classes = load_arxiv(torch.device("cpu"))
        exp = random_regular_expander_edges(n, args.exp_deg, seed=args.seed)
        if args.grid_auto:
            args.grid = auto_grid(world, n, ei.shape[1], args, rank, x.shape[1])
        if args.head_auto:
            args.grid = auto_head_grid(world, n, ei.shape[1], args, rank, x.shape[1])
            head_grid, args.fused = len(args.grid) == 3, len(args.grid) == 2
        nccl_tune(args, args.grid, n, world, rank, num_edges=ei.shape[1])
        grid = make_grid(n)
        gcn_blk, _ = grid.local_block(gcn_edges_with_self_loops(ei, n))
        if args.model == "exphormer":
            attn_ei, attn_et = ExphormerRef.build_attention_edges(ei, exp)
        else:                                                    # 1-hop mask (+ global token): no expander edges
            from gt3d.models.gt_variants import typed_graph_edges, graphormer_edges
            attn_ei, attn_et = (typed_graph_edges if args.model == "unimp" else graphormer_edges)(ei, n)
        attn_blk, attn_blk_et = grid.local_block(attn_ei, attn_et)
        x_loc = grid.take_owned(x).to(dev)
        y_loc = grid.take_owned(y).to(dev)
        masks = {}
        for name in ("train", "valid", "test"):
            m = torch.zeros(n, dtype=torch.bool); m[split[name]] = True
            masks[name] = grid.take_owned(m.to(torch.uint8)).bool().to(dev) & grid.valid_mask
        n_train = int(len(split["train"]))
        n_feat, e_total = x.shape[1], ei.shape[1]
    else:
        import os as _os
        plexus_fmt = _os.path.exists(_os.path.join(args.data, "metadata.pt"))       # Plexus partition_graph_2d output
        if plexus_fmt:
            from gt3d.data.plexus_loader import load_plexus_partition as load_partition, read_plexus_meta, grid_n_pad
            meta = read_plexus_meta(args.data)
            meta["n_pad"] = grid_n_pad(meta, world, 4096)
        else:
            from gt3d.data.loader import load_partition, read_meta
            meta = read_meta(args.data)
        n, num_classes, n_feat = meta["num_nodes"], meta["num_classes"], meta["num_features"]
        if args.grid_auto:
            e_for_model = meta["num_edges"] if meta.get("num_edges") else None
            if e_for_model is None:
                if rank == 0:
                    print("[grid_auto] Plexus partition has no edge count in its metadata: counting the local blocks first", flush=True)
            args.grid = auto_grid(world, n, e_for_model, args, rank, n_feat) if e_for_model else args.grid
        if args.head_auto:
            assert meta.get("num_edges"), "--head_auto needs the edge count in the partition metadata"
            args.grid = auto_head_grid(world, n, meta["num_edges"], args, rank, n_feat)
            head_grid, args.fused = len(args.grid) == 3, len(args.grid) == 2
        nccl_tune(args, args.grid, n, world, rank, num_edges=meta.get("num_edges"))
        grid = make_grid(n, n_pad=meta["n_pad"])
        part = load_partition(args.data, grid, dev)
        if meta.get("num_edges") is None:                                          # Plexus metadata has no edge count
            cnt = torch.tensor([part["num_local_edges"]], device=dev); dist.all_reduce(cnt); meta["num_edges"] = int(cnt.item())
        x_loc, y_loc, masks = part["x"], part["y"], part["masks"]
        gcn_blk, gcn_w = part["edge_index"], part["weight"]                # original edges incl. self loops, GCN weights
        # attention edges: the same edges (type 0) plus expander edges (type 1) generated block-wise
        # from hash-based permutations (no global edge list, no communication)
        if args.model == "exphormer":
            from gt3d.data.expander import block_expander_edges
            exp_blk = block_expander_edges(grid, n, args.exp_deg, seed=args.seed, device=dev)
            attn_blk = torch.cat([gcn_blk, exp_blk], dim=1)
            attn_blk_et = torch.cat([torch.zeros(gcn_blk.shape[1], dtype=torch.long, device=dev),
                                     torch.ones(exp_blk.shape[1], dtype=torch.long, device=dev)])
        else:
            # the graph's edges incl. self loops (type 1 = self loop) and, for Graphormer, the global token's edges
            from gt3d.models.gt_variants import type_local_block, token_edges_local
            exp_blk = None
            attn_blk = gcn_blk.to(dev)
            attn_blk_et = type_local_block(grid, attn_blk, n)
            if args.model == "graphormer":
                tok_ei, tok_et = token_edges_local(grid, n, dev)
                attn_blk, attn_blk_et = torch.cat([attn_blk, tok_ei], dim=1), torch.cat([attn_blk_et, tok_et])
        cnt = torch.tensor([int(masks["train"].sum())], device=dev); dist.all_reduce(cnt); n_train = int(cnt.item())
        e_total = meta["num_edges"]
    if args.auto_ckpt and not args.checkpoint:
        from gt3d.dist.grid_select import peak_memory_bytes, peak_memory_head_bytes, attention_edges, GB
        e_attn = attention_edges(n, e_total, args.exp_deg)
        if head_grid:
            pred = peak_memory_head_bytes(args.grid[0], args.grid[1], args.grid[2], n, e_attn, args.hidden, n_feat, args.layers,
                                          heads=args.heads, kernel_bf16=args.kernel_bf16, with_gcn=not args.no_gcn, e_gcn=e_total + n) / GB
        else:
            pred = peak_memory_bytes(args.grid[0], args.grid[1], n, e_attn, e_total + n, args.hidden, n_feat, args.layers,
                                     max(args.stream_sub, 1), args.stream_sub > 1, args.rs_bf16, layer_ckpt=False, heads=args.heads) / GB
        if pred > 34.0:
            args.checkpoint = True
        if rank == 0:
            print(f"[auto_ckpt] predicted peak without checkpointing {pred:.1f} GB -> checkpoint={'on' if args.checkpoint else 'off'}", flush=True)
    def mem(tag, model=None):
        if rank == 0:
            s = (f"[mem] {tag:28s} allocated {torch.cuda.memory_allocated()/2**30:6.2f} GB  peak {torch.cuda.max_memory_allocated()/2**30:6.2f} GB"
                 f"  reserved {torch.cuda.memory_reserved()/2**30:6.2f} GB")
            if model is not None:
                s += f"  graph structures {model.structure_bytes()/2**30:5.2f} GB"
            print(s, flush=True)
    mem("after data")
    if rank == 0:
        print(f"data: N={n} (padded {grid.n_pad}, chunk {grid.chunk}) grid={tuple(args.grid)} backend={args.backend} "
              f"E={e_total} local blocks: gcn {gcn_blk.shape[1]} attn {attn_blk.shape[1]} n_train={n_train}", flush=True)

    degree_loc = token_row = None
    if args.model == "graphormer":
        from gt3d.models.gt_variants import owned_in_degree, token_owned_row
        degree_loc = owned_in_degree(grid, attn_blk.to(dev), attn_blk_et.to(dev), dev)
        token_row = token_owned_row(grid, n)
    torch.manual_seed(args.seed)
    if args.model != "exphormer":
        assert not args.sparse_comm, "--sparse_comm is implemented for the Exphormer layers"
        from gt3d.models.gt_variants import DistGTVariant
        model = DistGTVariant(grid, attn_blk.to(dev), attn_blk_et.to(dev), n_feat, args.hidden, num_classes, args.layers, args.heads,
                              model=args.model, dropout=args.dropout, backend=args.backend,
                              comm_dtype=torch.bfloat16 if args.comm_bf16 else None, rs_dtype=torch.bfloat16 if args.rs_bf16 else None,
                              checkpoint=args.checkpoint, num_classes_label=num_classes if args.label_input else None,
                              max_degree=args.max_degree, token_row=token_row, ffn_mult=args.ffn_mult).to(dev)
    else:
      model = DistExphormer(grid, gcn_blk.to(dev), attn_blk.to(dev), attn_blk_et.to(dev), n_feat, args.hidden,
                          num_classes, args.layers, args.heads, dropout=args.dropout, gnn_dropout=args.dropout,
                          backend=args.backend, gcn_weight=gcn_w, checkpoint=args.checkpoint,
                          comm_dtype=torch.bfloat16 if args.comm_bf16 else None, fused=args.fused,
                          rs_dtype=torch.bfloat16 if args.rs_bf16 else None, pipeline_sub=args.pipeline_sub, stream_sub=args.stream_sub, norm=args.norm, ring=args.ring, with_gcn=not args.no_gcn, with_attn=not args.no_attn, gather_proj=args.gather_proj, stream_overlap=args.stream_overlap, stream_compact=not args.no_stream_compact,
                          kernel_dtype=torch.bfloat16 if args.kernel_bf16 else None,
                          sparse_comm=args.sparse_comm, sparse_threshold=(args.sparse_threshold if args.sparse_threshold_row is None
                                                                            else (args.sparse_threshold_row, args.sparse_threshold))).to(dev)
    if args.sparse_comm and rank == 0:
        inf = model.sparse_info or {}
        print(f"[sparse_comm] row group: rows needed {inf.get('row_rows')} / {inf.get('n_row_block')} share {inf.get('row_share', 0):.3f} "
              f"(a2a/AG {inf.get('row_ratio', 0):.2f}) -> {'SPARSE' if inf.get('use_row') else 'dense'};  "
              f"column group: {inf.get('col_rows')} / {inf.get('n_col_block')} share {inf.get('col_share', 0):.3f} "
              f"(a2a/AG {inf.get('col_ratio', 0):.2f}) -> {'SPARSE' if inf.get('use_col') else 'dense'}", flush=True)
    # make sure every rank starts from identical parameters
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    if args.compile:
        for i, layer in enumerate(model.layers):
            model.layers[i] = torch.compile(layer, dynamic=False)
    params = list(model.parameters())
    del gcn_blk, attn_blk, attn_blk_et                              # the model holds its own CSR/CSC structures
    if args.data is not None:
        del part, exp_blk

    def run_model(train):
        """Forward of the selected model; for --label_input (train mode) also returns the loss mask / count."""
        if args.model == "exphormer":
            return model(x_loc), masks["train"], n_train
        label_feat, loss_mask, n_loss = None, masks["train"], n_train
        if args.label_input:
            vis = masks["train"] & (torch.rand(y_loc.shape[0], device=dev) < args.label_rate) if train else masks["train"]
            label_feat = torch.where(vis, y_loc, torch.full_like(y_loc, num_classes))
            if train:
                loss_mask = masks["train"] & ~vis
                cnt = torch.tensor([int(loss_mask.sum())], device=dev); dist.all_reduce(cnt); n_loss = int(cnt.item())
        return model(x_loc, label_feat, degree_loc), loss_mask, n_loss
    gcn_w = None
    torch.cuda.empty_cache()
    mem("after model build", model)
    if hasattr(model, "compaction_stats"):
        tot, ne = model.compaction_stats()
        if rank == 0 and tot > 0:
            print(f"[compact] attention (row, sub-block) pairs with edges: {ne}/{tot} = {100 * ne / tot:.1f}% "
                  f"(the rows the compact-accumulate kernels tile)", flush=True)
    cstats = _coll_stats(rank, world) if args.coll_stats else None
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, args.warmup, args.epochs))

    best = {"val": -1.0, "test": 0.0, "epoch": -1}
    t_train = t_eval = 0.0
    epoch_times = []
    cpu_times = []
    torch.cuda.reset_peak_memory_stats()
    import contextlib
    for epoch in range(args.epochs):
        torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        model.train()
        set_step(epoch)
        prof_ctx = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) \
            if (args.profile and epoch == 2) else contextlib.nullcontext()
        with prof_ctx as prof:
            if cstats is not None:
                cstats.rec = []                                  # drop the previous epoch's eval calls
            opt.zero_grad(set_to_none=True)
            if epoch == 2:
                torch.cuda.reset_peak_memory_stats()             # steady-state peak: after Triton autotuning (epochs 0-1)
                if args.timers:                                  # fresh AxoNN timers: exclude the autotuning epochs from the tree
                    import axonn.timers
                    ax.timers = axonn.timers.Timers()
            tm = ax.get_timers()
            tm.start("forward")
            if args.saved_census and epoch == 2:
                with _saved_census(grid, model, rank):
                    logits, loss_mask, n_loss = run_model(train=True)
            else:
                logits, loss_mask, n_loss = run_model(train=True)
            tm.stop("forward")
            if epoch in (0, 2): mem(f"after forward (epoch {epoch})", model)
            loss = dist_cross_entropy(logits, y_loc, loss_mask, n_loss)
            tm.start("backward")
            t_cpu0 = time.perf_counter()
            if cstats is not None:
                cstats.in_backward = True
            loss.backward()
            if cstats is not None:
                cstats.in_backward = False
            cpu_bwd_ms = (time.perf_counter() - t_cpu0) * 1e3            # CPU-side enqueue time of the backward
            tm.stop("backward")
            if epoch in (0, 2): mem(f"after backward (epoch {epoch})", model)
            tm.start("allreduce grads")
            allreduce_grads(params)
            tm.stop("allreduce grads")
            # clip on the GLOBAL gradient norm (identical on every rank after the all-reduce)
            tm.start("optimizer")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tm.stop("optimizer")
            torch.cuda.synchronize()
        if args.profile and epoch == 2:
            _print_profile(prof, rank, world)
            trace = os.environ.get("GT3D_PROF_TRACE")
            if trace and rank in (0, world // 2, world - 1):
                prof.export_chrome_trace(f"{trace}_rank{rank}.json")
        if args.timers and epoch >= 2:
            cpu_times.append(cpu_bwd_ms)
        torch.cuda.synchronize(); t1 = time.perf_counter(); t_train += t1 - t0
        epoch_times.append(t1 - t0)
        if cstats is not None:
            cstats.end_epoch(keep=epoch >= 2)
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = dist_accuracy(run_model(train=False)[0], y_loc, masks)
            torch.cuda.synchronize(); t_eval += time.perf_counter() - t1
            if acc["valid"] > best["val"]:
                best = {"val": acc["valid"], "test": acc["test"], "epoch": epoch}
            if rank == 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
                loss_tot = loss.detach().clone(); dist.all_reduce(loss_tot)
                print(f"epoch {epoch:4d} loss {loss_tot.item():.4f} train {acc['train']:.4f} val {acc['valid']:.4f} "
                      f"test {acc['test']:.4f} | best val {best['val']:.4f} test {best['test']:.4f} @ {best['epoch']} "
                      f"| {t_train/(epoch+1)*1e3:.1f} ms/epoch train", flush=True)
            elif rank != 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
                loss_tot = loss.detach().clone(); dist.all_reduce(loss_tot)
    if cstats is not None:
        cstats.report()
    if args.timers:
        # CPU-side backward enqueue time per rank (ms per epoch, epochs 2+): close to the GPU backward time = the
        # backward is paced by the CPU (launch / autograd bound); the spread over ranks is the skew the collectives absorb
        t_cpu = torch.tensor([sum(cpu_times) / max(len(cpu_times), 1)], device=dev)
        all_cpu = [torch.zeros_like(t_cpu) for _ in range(world)]
        dist.all_gather(all_cpu, t_cpu)
        if rank == 0:
            vals = [float(v.item()) for v in all_cpu]
            print(f"[cpu] backward enqueue ms per epoch: min {min(vals):.1f} max {max(vals):.1f} mean {sum(vals)/len(vals):.1f} | per rank: "
                  + " ".join(f"{v:.0f}" for v in vals), flush=True)
        times, counts = ax.get_timers().get_times()
        n_ep = max(1, len(epoch_times) - 2)
        per_epoch = {k: v / n_ep for k, v in times.items()}                     # ms per epoch over epochs 2+ (timers reset at epoch 2)
        if rank == 0:
            print(f"[timers] ms per epoch over {n_ep} steady epochs; max over ranks | mean | imbalance:", flush=True)
        print_axonn_timer_data(per_epoch)
    if rank == 0:
        print(f"FINAL model={args.model} label_input={int(args.label_input)} data={args.data or 'arxiv'} grid={tuple(args.grid)} gcn={int(not args.no_gcn)} attn={int(not args.no_attn)} ring={int(args.ring)} norm={args.norm} backend={args.backend} ckpt={args.checkpoint} fused={args.fused} compile={args.compile} comm_bf16={args.comm_bf16} rs_bf16={args.rs_bf16} tf32={int(args.tf32_matmul)} hidden={args.hidden} heads={args.heads} layers={args.layers} "
              f"best_val={best['val']:.4f} test@best_val={best['test']:.4f} "
              f"epoch={best['epoch']} train_time={t_train:.1f}s ({t_train/args.epochs*1e3:.1f} ms/epoch; "
              f"steady {1e3*sum(epoch_times[2:])/max(len(epoch_times)-2,1):.1f} ms/epoch over epochs 2+) eval_time={t_eval:.1f}s "
              f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
        print(f"STEADY model={args.model} grid={'x'.join(map(str, args.grid))} gcn={int(not args.no_gcn)} ring={int(args.ring)} hidden={args.hidden} heads={args.heads} fused={int(args.fused)} gproj={int(args.gather_proj)} bf16={int(args.comm_bf16)} rs_bf16={int(args.rs_bf16)} pipe={args.pipeline_sub} stream={args.stream_sub} sov={int(args.stream_overlap)} cmp={int(not args.no_stream_compact)} tf32={int(args.tf32_matmul)} kbf16={int(args.kernel_bf16)} hpp={os.environ.get('GT3D_HEADS_PER_PROGRAM', '4')} sparse={_sparse_tag(args, model)} norm={args.norm} layout={args.grid_layout} "
              f"ckpt={int(args.checkpoint)} ms_per_epoch={1e3*sum(epoch_times[2:])/max(len(epoch_times)-2,1):.1f} "
              f"peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f} (steady, after autotuning)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
