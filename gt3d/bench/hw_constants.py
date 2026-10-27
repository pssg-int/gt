"""Hardware / kernel constants for the white-box performance model (gt3d/dist/grid_select.py).

Nothing here is fitted to training runs: every number is a direct micro-measurement of one resource on the
machine the model is used on, in the shape the model needs it:

  hbm            : HBM stream rate (GB/s) of a 1 GB copy / elementwise op
  launch_us      : cost of one tiny kernel launch (us), i.e. the per-op floor of launch-bound segments
  ew             : effective rate (GB/s) of the elementwise ops on owned-row tensors of a given size
  gemm           : TFLOPS of the three GEMM shapes of a Linear on [M, d] rows (forward x W^T, grad_input, grad_W),
                   IEEE fp32 and TF32, per hidden size
  attn           : the fused attention kernels on a synthetic CSR block: ns per edge / per row and the achieved
                   bandwidth under the byte model used by grid_select (forward, backward)
  spmm           : cuSPARSE CSR SpMM (the GCN branch) on the same blocks: achieved bandwidth under its byte model
  coll           : NCCL all-gather / reduce-scatter (bf16) per-rank effective bandwidth (GB/s) and latency (ms)
                   for an intra-node group (4 consecutive ranks), the inter-node group (one rank per node) and the
                   intra-node all-to-all of the head axis, at the per-rank chunk sizes of the real grids

Launch (one process per GPU):
    GT3D_SCRIPT=bench/hw_constants.py srun --jobid=<id> -N 4 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh
Rank 0 prints the measurements and a `HW = {...}` block to paste into grid_select.py.
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention import triton_kernel as tk  # noqa: E402
from gt3d.attention.backend_torch import build_csr  # noqa: E402

GPN = 4


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3          # ms


# HW_ISOLATED (default 0): 0 = `iters` back-to-back calls, time / iters (pipelined launches); 1 = every call host-synchronised
# before and after (torch.cuda.synchronize, like grid2d._pre_collective in training) and timed on its own, mean over the
# calls; 2 = additionally a world barrier before every call (all groups start in lockstep).  Always max over the ranks.
ISOLATED = int(os.environ.get("HW_ISOLATED", "0"))
# HW_WARMUP_MS (isolated modes): run ~this many ms of matmuls on the stream before every timed call (the GPU arrives hot and
# busy, as in training); HW_FRESH=1: the output tensor of every call is a fresh torch.empty (as in training) instead of a
# reused buffer.
WARMUP_MS = float(os.environ.get("HW_WARMUP_MS", "0"))
FRESH = int(os.environ.get("HW_FRESH", "0"))
TOUCH = int(os.environ.get("HW_TOUCH", "0"))          # rewrite the input buffer with a kernel before every timed call
_WARM = {}
_TOUCH = {}


def _busy(ms):
    if ms <= 0:
        return
    if "a" not in _WARM:
        _WARM["a"] = torch.randn(4096, 4096, device="cuda"); _WARM["b"] = torch.empty_like(_WARM["a"])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            torch.matmul(_WARM["a"], _WARM["a"], out=_WARM["b"])
        torch.cuda.synchronize(); _WARM["per"] = (time.perf_counter() - t0) / 10 * 1e3
    for _ in range(max(1, int(ms / _WARM["per"]))):
        torch.matmul(_WARM["a"], _WARM["a"], out=_WARM["b"])


def coll_time(fn, group, iters=10, warm=3):
    """ms per call, max over the ranks of the whole world (every group runs concurrently, as in training)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    if ISOLATED == 0:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
    else:
        ts = []
        for _ in range(iters):
            _busy(WARMUP_MS)
            if TOUCH and _TOUCH.get("buf") is not None:
                _TOUCH["buf"].mul_(1.0)
            torch.cuda.synchronize()
            if ISOLATED >= 2:
                dist.barrier()                                   # after the local work: every rank starts the call together
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ms = sum(ts) / len(ts) * 1e3
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main():
    only = os.environ.get("HW_ONLY", "")                        # e.g. HW_ONLY=nccl: collectives only
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = rank % torch.cuda.device_count()
    torch.cuda.set_device(local)
    dev = torch.device("cuda")
    nodes = world // GPN
    out = {}
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    say(f"[hw] world={world} nodes={nodes} device={torch.cuda.get_device_name()}")

    if only in ("", "compute"):
        compute_part(out, dev, say)
    if only in ("", "nccl"):
        nccl_part(out, dev, say, rank, world, nodes)
    if only == "dual":
        dual_part(out, dev, say, rank, world, nodes)
    if rank == 0:
        tag = os.environ.get("HW_TAG", "")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hw_constants{'_' + only if only else ''}{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        say(f"[hw] written {path}")
    dist.barrier()
    dist.destroy_process_group()


def compute_part(out, dev, say):
    # ---------------------------------------------------------------- HBM / elementwise / launch
    x = torch.empty(2 ** 28, device=dev); y = torch.empty_like(x)       # 1 GB fp32 each
    t = timeit(lambda: y.copy_(x))
    out["hbm_copy_gbs"] = 2 * x.numel() * 4 / t / 1e6
    t = timeit(lambda: torch.mul(x, 2.0, out=y))
    out["hbm_mul_gbs"] = 2 * x.numel() * 4 / t / 1e6
    say(f"[hw] HBM copy {out['hbm_copy_gbs']:.0f} GB/s, mul {out['hbm_mul_gbs']:.0f} GB/s")
    del x, y
    tiny = torch.zeros(1, device=dev)
    t = timeit(lambda: [tiny.add_(1.0) for _ in range(200)], iters=5)
    out["launch_us"] = t / 200 * 1e3
    say(f"[hw] tiny kernel launch {out['launch_us']:.1f} us")
    # owned-row elementwise segment of the layer (LN fwd+bwd, dropout, residual adds, relu) on [O, d]
    out["ew"] = {}
    for rows in (3_648, 38_272, 153_088, 612_352):
        for d in (80, 128, 256):
            a = torch.randn(rows, d, device=dev, requires_grad=True)
            ln = torch.nn.LayerNorm(d).to(dev)

            def seg():
                h = ln(a + torch.nn.functional.dropout(a, 0.3, True))
                h = torch.relu(h) + a
                h.sum().backward()
            t = timeit(seg)
            nbytes = 14 * rows * d * 4                              # ~14 passes over [rows, d] fp32 (fwd + bwd)
            out["ew"][f"{rows}x{d}"] = dict(ms=t, gbs=nbytes / t / 1e6, ops=11)
            del a, ln
    say("[hw] elementwise segment (LN + dropout + relu + 2 adds, fwd+bwd) ms / effective GB/s over 14 passes:")
    for k, v in out["ew"].items():
        say(f"[hw]   {k:>12s}: {v['ms']:6.3f} ms  {v['gbs']:6.0f} GB/s")

    # ---------------------------------------------------------------- GEMM
    out["gemm"] = {}
    M = 612_352
    for d in (64, 80, 128, 256):
        a = torch.randn(M, d, device=dev); w = torch.randn(d, d, device=dev); g = torch.randn(M, d, device=dev)
        for prec in ("ieee", "tf32"):
            torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
            flops = 2.0 * M * d * d
            t_f = timeit(lambda: a @ w.t())                        # forward x W^T
            t_i = timeit(lambda: g @ w)                            # grad_input
            t_w = timeit(lambda: g.t() @ a)                        # grad_W  (reduction over M)
            out["gemm"][f"{prec}_{d}"] = dict(fwd=flops / t_f / 1e9, gin=flops / t_i / 1e9, gw=flops / t_w / 1e9)
            say(f"[hw] GEMM {prec} d={d:3d} M={M}: fwd {flops/t_f/1e9:6.1f}  grad_in {flops/t_i/1e9:6.1f}  grad_W {flops/t_w/1e9:6.1f} TFLOPS")
        del a, w, g
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---------------------------------------------------------------- attention kernels + SpMM on synthetic blocks
    # (rows R, cols C, edges E) ~ products at 64 GPUs 16x4 / 16 GPUs 4x4 / 4 GPUs 2x2, Reddit at 64 GPUs, a streamed sub-block
    out["attn"], out["spmm"] = {}, {}
    gen = torch.Generator(device=dev).manual_seed(0)
    blocks = [("products64", 153_088, 612_352, 1_976_184), ("products16", 612_352, 612_352, 8_837_500),
              ("products4", 1_224_704, 1_224_704, 35_350_000), ("reddit64", 14_592, 58_368, 1_779_295),
              ("stream_sub", 1_735_360, 27_120, 500_000)]
    for name, R, C, E in blocks:
        src = torch.randint(0, C, (E,), device=dev, generator=gen)
        dst = torch.randint(0, R, (E,), device=dev, generator=gen)
        et = torch.randint(0, 2, (E,), device=dev, generator=gen)
        crow, col, etype, _ = build_csr(torch.stack([src, dst]), R, et)
        crow32, col32, etype32 = crow.to(torch.int32).contiguous(), col.to(torch.int32), etype.to(torch.int32)
        for h, dh in ((2, 40), (4, 32), (8, 32)):
            d = h * dh
            q = torch.randn(R, h, dh, device=dev); k = torch.randn(C, h, dh, device=dev); v = torch.randn(C, h, dh, device=dev)
            gate = torch.randn(2, h, dh, device=dev); dacc = torch.randn(R, h, dh, device=dev); dl = torch.randn(R, h, device=dev)
            m0 = torch.zeros(R, h, device=dev)
            t_f = timeit(lambda: tk.attn_fwd_partial(q, k, v, crow32, col32, etype32, gate, None, score="gated", norm="clampexp"))
            t_b = timeit(lambda: tk.attn_bwd_partial(q, k, v, crow32, col32, etype32, gate, None, m0, dacc, dl, score="gated", norm="clampexp"))
            # byte model of grid_select: fwd E (2 d 4 + 5) + R (3 d 4); bwd row pass E (2 d 4 + 5) + R (4 d 4), col pass E (2 d 4 + 5 + 8 h) + C (4 d 4)
            b_f = E * (8 * d + 5) + R * 12 * d
            b_b = E * (16 * d + 10 + 8 * h) + R * 16 * d + C * 16 * d
            tiled = tk._use_tiled(crow32, col32)
            out["attn"][f"{name}_h{h}d{dh}"] = dict(R=R, C=C, E=E, fwd_ms=t_f, bwd_ms=t_b, fwd_gbs=b_f / t_f / 1e6, bwd_gbs=b_b / t_b / 1e6,
                                                     fwd_ns_edge=t_f * 1e6 / E, bwd_ns_edge=t_b * 1e6 / E, tiled=bool(tiled))
            say(f"[hw] attn {name:11s} h{h} d{dh} ({E/R:5.1f} edges/row, {'tiled' if tiled else 'per-row'}): fwd {t_f:7.2f} ms ({b_f/t_f/1e6:5.0f} GB/s, "
                f"{t_f*1e6/E:5.2f} ns/edge, {t_f*1e6/R:5.1f} ns/row)  bwd {t_b:7.2f} ms ({b_b/t_b/1e6:5.0f} GB/s, {t_b*1e6/E:5.2f} ns/edge, {t_b*1e6/R:5.1f} ns/row)")
            # cuSPARSE SpMM (GCN): A [R, C] @ x [C, d]  and the transpose A^T [C, R] @ g [R, d]
            w = torch.rand(E, device=dev)
            a = torch.sparse_csr_tensor(crow32, col32, w, size=(R, C))
            perm_t = torch.argsort(col, stable=True)
            ccol = torch.zeros(C + 1, dtype=torch.long, device=dev); ccol[1:] = torch.cumsum(torch.bincount(col, minlength=C), 0)
            rows = torch.repeat_interleave(torch.arange(R, device=dev), crow[1:] - crow[:-1])
            at = torch.sparse_csr_tensor(ccol.to(torch.int32), rows[perm_t].to(torch.int32), w[perm_t], size=(C, R))
            xc = torch.randn(C, d, device=dev); gr = torch.randn(R, d, device=dev)
            t_s = timeit(lambda: torch.sparse.mm(a, xc)); t_st = timeit(lambda: torch.sparse.mm(at, gr))
            b_s = E * (4 * d + 8) + (R + C) * 4 * d
            out["spmm"][f"{name}_d{d}"] = dict(fwd_ms=t_s, bwd_ms=t_st, fwd_gbs=b_s / t_s / 1e6, bwd_gbs=b_s / t_st / 1e6)
            say(f"[hw] spmm {name:11s} d={d:3d}: A x {t_s:6.2f} ms ({b_s/t_s/1e6:5.0f} GB/s)  A^T g {t_st:6.2f} ms ({b_s/t_st/1e6:5.0f} GB/s)")
            del q, k, v, gate, dacc, dl, m0, w, a, at, xc, gr
        del src, dst, et, crow, col, etype, crow32, col32, etype32
        torch.cuda.empty_cache()



def dual_part(out, dev, say, rank, world, nodes):
    """Two cross-node groups per rank, alternating (the (2,2,4) / (4,4,4) situation): node n = (r, c) on a 2 x 2 node
    grid, local rank h; group A(r, h) = the two nodes of row r, group B(c, h) = the two nodes of column c (one rank per
    node each).  Measures AG on A alone, AG A / AG B alternating, and alternating with a ~5 ms matmul between, per call."""
    assert nodes == 4 and world == 16, "dual_part needs 4 nodes x 4 GPUs"
    n, h = rank // GPN, rank % GPN
    r, c = n // 2, n % 2
    groups_a = {(rr, hh): dist.new_group([(2 * rr + cc) * GPN + hh for cc in range(2)]) for rr in range(2) for hh in range(GPN)}
    groups_b = {(cc, hh): dist.new_group([(2 * rr + cc) * GPN + hh for rr in range(2)]) for cc in range(2) for hh in range(GPN)}
    ga, gb = groups_a[(r, h)], groups_b[(c, h)]
    big = torch.randn(4096, 4096, device=dev)
    out["dual"] = {}
    sizes = [int(v) for v in os.environ.get("HW_DUAL_SIZES", "16,32,64,128").split(",")]
    for mb in sizes:
        n_el = int(mb * 2 ** 20 / 2)
        chunk = torch.randn(n_el, device=dev).to(torch.bfloat16)
        fa = torch.empty(2 * n_el, device=dev, dtype=torch.bfloat16); fb = torch.empty_like(fa)
        def timed(fn, iters=10, warm=3):
            for _ in range(warm):
                fn()
            torch.cuda.synchronize(); dist.barrier()
            evs = []
            for _ in range(iters):
                s_ = torch.cuda.Event(enable_timing=True); e_ = torch.cuda.Event(enable_timing=True)
                s_.record(); fn(); e_.record(); evs.append((s_, e_))
            torch.cuda.synchronize()
            t = torch.tensor([sum(s_.elapsed_time(e_) for s_, e_ in evs) / iters], device=dev)
            tmax = t.clone(); dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
            tmean = t.clone(); dist.all_reduce(tmean); tmean /= world
            return float(tmax.item()), float(tmean.item())
        res = {}
        res["A_alone"] = timed(lambda: dist.all_gather_into_tensor(fa, chunk, group=ga))
        res["B_alone"] = timed(lambda: dist.all_gather_into_tensor(fb, chunk, group=gb))
        res["A_then_B"] = timed(lambda: (dist.all_gather_into_tensor(fa, chunk, group=ga), dist.all_gather_into_tensor(fb, chunk, group=gb)))
        res["A_A_B_B"] = timed(lambda: (dist.all_gather_into_tensor(fa, chunk, group=ga), dist.all_gather_into_tensor(fa, chunk, group=ga),
                                        dist.all_gather_into_tensor(fb, chunk, group=gb), dist.all_gather_into_tensor(fb, chunk, group=gb)))
        res["A_sync_B"] = timed(lambda: (dist.all_gather_into_tensor(fa, chunk, group=ga), torch.cuda.synchronize(), dist.barrier(),
                                         dist.all_gather_into_tensor(fb, chunk, group=gb), torch.cuda.synchronize(), dist.barrier()))
        res["A_mm_B_mm"] = timed(lambda: (dist.all_gather_into_tensor(fa, chunk, group=ga), big @ big, dist.all_gather_into_tensor(fb, chunk, group=gb), big @ big))
        res["mm_only"] = timed(lambda: (big @ big, big @ big))
        out["dual"][mb] = res
        say(f"[hw] dual {mb} MB chunk (AG over 2 ranks on 2 nodes, received {mb} MB): " +
            "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f} ms" for k, v in res.items()) + "   (max / mean over ranks per call sequence)")
        del chunk, fa, fb
    del big


def nccl_part(out, dev, say, rank, world, nodes):
    # ---------------------------------------------------------------- NCCL collectives
    # group layouts "k x m": k ranks per node on m nodes (the row / column / head groups of the grids are all of this
    # form: k consecutive local ranks, nodes with a fixed stride).  All groups of one layout run concurrently (every
    # rank is in exactly one), as in training.  Per-rank received bytes / time = the effective rate of the layout.
    # HW_LAYOUTS="1x16,4x16" restricts the layouts (default: every k in {1,2,4} x m in {1,2,4,8,16} with m <= nodes);
    # HW_SIZES="1,9,64" the per-rank chunk sizes in MB; HW_TAG a suffix of the output file.
    out["coll"] = {}
    out["isolated"] = ISOLATED
    want = os.environ.get("HW_LAYOUTS", "")
    want = set(want.split(",")) if want else None
    groups = {}
    for k in (1, 2, 4):
        for m in (1, 2, 4, 8, 16):
            if k * m == 1 or m > nodes or (want is not None and f"{k}x{m}" not in want):
                continue
            node_sets = [list(range(a, a + m)) for a in range(0, nodes, m)]
            local_sets = [list(range(a, a + k)) for a in range(0, GPN, k)]
            mine = None
            for ns in node_sets:
                for ls in local_sets:
                    g = dist.new_group([n * GPN + i for n in ns for i in ls])
                    if rank // GPN in ns and rank % GPN in ls:
                        mine = g
            groups[f"{k}x{m}"] = mine
    sizes_mb = [float(v) for v in os.environ.get("HW_SIZES", "0.25,1,4,16,64,160").split(",")]   # per-rank chunk (bf16)
    for gname, group in groups.items():
        g = dist.get_world_size(group)
        res = {}
        for mb in sizes_mb:
            n_el = int(mb * 2 ** 20 / 2)
            chunk = torch.randn(n_el, device=dev).to(torch.bfloat16)
            full = torch.empty(g * n_el, device=dev, dtype=torch.bfloat16)
            _TOUCH["buf"] = full
            if FRESH:
                t_ag = coll_time(lambda: dist.all_gather_into_tensor(torch.empty_like(full), chunk, group=group), group)
                t_rs = coll_time(lambda: dist.reduce_scatter_tensor(torch.empty_like(chunk), full, group=group), group)
            else:
                t_ag = coll_time(lambda: dist.all_gather_into_tensor(full, chunk, group=group), group)
                t_rs = coll_time(lambda: dist.reduce_scatter_tensor(chunk, full, group=group), group)
            recv = (g - 1) * n_el * 2
            res[mb] = dict(ag_ms=t_ag, rs_ms=t_rs, ag_gbs=recv / t_ag / 1e6, rs_gbs=recv / t_rs / 1e6)
            t_a2a = coll_time(lambda: dist.all_to_all_single(full[:n_el], chunk, group=group), group)
            res[mb]["a2a_ms"] = t_a2a; res[mb]["a2a_gbs"] = (g - 1) / g * n_el * 2 / t_a2a / 1e6
            del chunk, full
        out["coll"][gname] = dict(size=g, by_mb=res)
        say(f"[hw] NCCL layout {gname} (group size {g}): per-rank chunk -> all-gather / reduce-scatter / all-to-all ms (effective GB/s per rank of received bytes)")
        for mb, r in res.items():
            say(f"[hw]   {mb:6.2f} MB: AG {r['ag_ms']:7.3f} ms ({r['ag_gbs']:5.1f} GB/s)  RS {r['rs_ms']:7.3f} ms ({r['rs_gbs']:5.1f} GB/s)"
                f"  A2A {r['a2a_ms']:7.3f} ms ({r['a2a_gbs']:5.1f} GB/s)")
        # alpha-beta: alpha from the smallest message, beta from the largest
        lo, hi = res[sizes_mb[0]], res[sizes_mb[-1]]                 # (equal for a single-size run)
        out["coll"][gname]["ag_alpha_ms"] = lo["ag_ms"]; out["coll"][gname]["ag_gbs"] = hi["ag_gbs"]
        out["coll"][gname]["rs_alpha_ms"] = lo["rs_ms"]; out["coll"][gname]["rs_gbs"] = hi["rs_gbs"]
        out["coll"][gname]["a2a_alpha_ms"] = lo["a2a_ms"]; out["coll"][gname]["a2a_gbs"] = hi["a2a_gbs"]


if __name__ == "__main__":
    main()
