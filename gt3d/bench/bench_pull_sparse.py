"""Receiver-driven ("pull") sparse collectives over NVLink with PyTorch symmetric memory, against NCCL and the
all-to-all-v plan of gt3d/dist/sparse_comm.py — one node, 4 GPUs.

The a2a-v path pays two memory passes that NCCL's dense collectives do not: the sender's index_select (packing the
scattered needed rows into per-receiver segments) and, for the reduce-scatter, the receiver's segmented sum (SpMM)
after the data has landed.  With every rank's buffer mapped into every peer's address space (symmetric memory, CUDA
IPC over NVLink) a receiver can instead
  * pull-gather : read exactly the rows it needs straight out of each peer's chunk into its compact buffer
                  (one index_select whose SOURCE is remote): no packing, no intermediate buffer;
  * pull-reduce : for every owned row, read that row's partial from every peer that holds it, sum in registers and
                  store once (a Triton kernel with the peers' compact buffers as pointers): the reduction is fused into
                  the transfer, no SpMM pass.
Synthetic need sets of density rho over the 4 x rows_per_rank block (same on every rank); correctness against the
a2a-v plan (same compact order), then timing of: NCCL all-gather / reduce-scatter (dense), plan.gather /
plan.reduce_scatter (a2a-v), pull-gather, pull-reduce.  ms = max over ranks of the mean of --iters calls, each call
preceded by a symmetric-memory barrier (part of the primitive's cost).

    GT3D_SCRIPT=bench/bench_pull_sparse.py srun -N1 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh --rows 612352 --dims 128 256 --densities 0.2 0.6 1.0
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.dist.sparse_comm import SparseRowPlan  # noqa: E402
from gt3d.dist.grid2d import raw_all_gather_rows, raw_reduce_scatter_rows  # noqa: E402


# ------------------------------------------------------------------------------------------------------------
# fused pull-reduce: out[i, :] = sum over peers s of P_s[idx[i, s], :]  (idx = -1: peer s holds no partial for row i)
# ------------------------------------------------------------------------------------------------------------
@triton.jit
def _pull_reduce_kernel(out_ptr, idx_ptr, p0, p1, p2, p3, n_rows, F, stride, out_stride,
                        G: tl.constexpr, BLOCK_F: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_F)
    rm = rows < n_rows
    cm = cols < F
    m2 = rm[:, None] & cm[None, :]
    acc = tl.zeros((ROWS, BLOCK_F), dtype=tl.float32)
    for s in tl.static_range(G):
        idx = tl.load(idx_ptr + rows * G + s, mask=rm, other=-1)
        ok = m2 & (idx[:, None] >= 0)
        off = idx[:, None].to(tl.int64) * stride + cols[None, :]
        if s == 0:
            v = tl.load(p0 + off, mask=ok, other=0.0)
        elif s == 1:
            v = tl.load(p1 + off, mask=ok, other=0.0)
        elif s == 2:
            v = tl.load(p2 + off, mask=ok, other=0.0)
        else:
            v = tl.load(p3 + off, mask=ok, other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * out_stride + cols[None, :], acc.to(out_ptr.dtype.element_ty), mask=m2)


# ------------------------------------------------------------------------------------------------------------
# pull-gather: out[off + j, :] = src[src_idx[j], :] for one peer's segment (src = the peer's mapped chunk)
# ------------------------------------------------------------------------------------------------------------
@triton.jit
def _pull_gather_kernel(out_ptr, src_ptr, idx_ptr, n, F, src_stride, out_stride, BLOCK_F: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_F)
    rm = rows < n
    cm = cols < F
    m2 = rm[:, None] & cm[None, :]
    idx = tl.load(idx_ptr + rows, mask=rm, other=0)
    v = tl.load(src_ptr + idx[:, None].to(tl.int64) * src_stride + cols[None, :], mask=m2, other=0.0)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * out_stride + cols[None, :], v, mask=m2)


def pull_gather_triton(out, peers, src_idx, offs, cnts):
    """out [n_need, F] compact buffer; for every peer s: out[offs[s]:offs[s]+cnts[s]] = peers[s][src_idx[s]] (remote read)."""
    F = out.shape[1]
    BLOCK_F = triton.next_power_of_2(F)
    ROWS = 8 if BLOCK_F <= 256 else 4
    for s in range(len(peers)):
        n = cnts[s]
        if n == 0:
            continue
        seg = out[offs[s]:offs[s] + n]
        _pull_gather_kernel[(triton.cdiv(n, ROWS),)](seg, peers[s], src_idx[s], n, F, peers[s].stride(0), seg.stride(0),
                                                     BLOCK_F=BLOCK_F, ROWS=ROWS, num_warps=4)
    return out


# single kernel over the whole compact buffer: row k reads peer src_rank[k] at row src_row[k]; the program -> row-block
# map is round-robin over G slices of the buffer so that at any moment programs are reading from ALL peers at once
# (the per-peer kernels above read one NVLink pair at a time: ~100 GB/s instead of the 3 x 100 GB/s of the node)
@triton.jit
def _pull_gather_mixed_kernel(out_ptr, rank_ptr, idx_ptr, p0, p1, p2, p3, n, n_blocks, F, src_stride, out_stride,
                              G: tl.constexpr, BLOCK_F: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    per = (n_blocks + G - 1) // G
    b = (pid % G) * per + pid // G                       # round-robin over G slices of the row blocks
    rows = b * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_F)
    rm = (rows < n) & (b < n_blocks)
    cm = cols < F
    m2 = rm[:, None] & cm[None, :]
    src = tl.load(rank_ptr + rows, mask=rm, other=-1)
    idx = tl.load(idx_ptr + rows, mask=rm, other=0)
    off = idx[:, None].to(tl.int64) * src_stride + cols[None, :]
    v = tl.zeros((ROWS, BLOCK_F), dtype=tl.float32)
    for s in tl.static_range(G):
        ok = m2 & (src[:, None] == s)
        if s == 0:
            v += tl.load(p0 + off, mask=ok, other=0.0).to(tl.float32)
        elif s == 1:
            v += tl.load(p1 + off, mask=ok, other=0.0).to(tl.float32)
        elif s == 2:
            v += tl.load(p2 + off, mask=ok, other=0.0).to(tl.float32)
        else:
            v += tl.load(p3 + off, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * out_stride + cols[None, :], v.to(out_ptr.dtype.element_ty), mask=m2)


def pull_gather_mixed(out, peers, src_rank, src_row):
    n, F = out.shape
    G = len(peers)
    ps = list(peers) + [peers[0]] * (4 - G)
    BLOCK_F = triton.next_power_of_2(F)
    ROWS = 8 if BLOCK_F <= 256 else 4
    n_blocks = triton.cdiv(n, ROWS)
    grid = (((n_blocks + G - 1) // G) * G,)
    _pull_gather_mixed_kernel[grid](out, src_rank, src_row, ps[0], ps[1], ps[2], ps[3], n, n_blocks, F, peers[0].stride(0), out.stride(0),
                                    G=G, BLOCK_F=BLOCK_F, ROWS=ROWS, num_warps=4)
    return out


def pull_reduce(out, idx, peers):
    """out [n_rows, F]; idx int32 [n_rows, G]; peers: G tensors [*, F] (the peers' compact partial buffers, mapped)."""
    n_rows, F = out.shape
    G = len(peers)
    assert G <= 4 and idx.shape == (n_rows, G)
    ps = list(peers) + [peers[0]] * (4 - G)
    BLOCK_F = triton.next_power_of_2(F)
    ROWS = 8 if BLOCK_F <= 256 else 4
    grid = (triton.cdiv(n_rows, ROWS),)
    _pull_reduce_kernel[grid](out, idx, ps[0], ps[1], ps[2], ps[3], n_rows, F, peers[0].stride(0), out.stride(0),
                              G=G, BLOCK_F=BLOCK_F, ROWS=ROWS, num_warps=4)
    return out


def timeit(fn, iters, hdl, warm=3):
    for _ in range(warm):
        hdl.barrier(channel=0); fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        hdl.barrier(channel=0); fn()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return float(ms.item())


def timeit_nccl(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return float(ms.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=612352, help="rows per rank (products at 4 GPUs: 612,352)")
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--densities", type=float, nargs="+", default=[0.2, 0.6, 1.0])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    group = dist.group.WORLD
    gname = group.group_name
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(gname)
    R = args.rows
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    say(f"[pull] torch {torch.__version__} symmetric memory on group {gname}, world {world}, rows/rank {R}")
    for dim in args.dims:
        F = dim
        # ---- symmetric buffers: my chunk (gather source) and my compact partial (reduce source) ----
        x = symm_mem.empty((R, F), dtype=torch.bfloat16, device=dev)
        hx = symm_mem.rendezvous(x, group=gname)
        x.normal_()
        for rho in args.densities:
            # need set of every rank (same density, independent random rows), known to all (one all_gather_object)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
            k = max(1, int(round(rho * world * R)))
            need = torch.sort(torch.randperm(world * R, generator=gen)[:k])[0]
            needs = [None] * world
            dist.all_gather_object(needs, need.tolist() if world * R <= 4 * 10 ** 6 else need.numpy())
            needs = [torch.as_tensor(n, dtype=torch.long) for n in needs]
            need_d = need.to(dev)
            plan = SparseRowPlan(group, R, need_d, name=f"rho{rho}")
            n_need = plan.n_need
            n_max = int(torch.tensor([n_need], device=dev).max().item())
            n_max_t = torch.tensor([n_max], device=dev); dist.all_reduce(n_max_t, op=dist.ReduceOp.MAX); n_max = int(n_max_t.item())
            # ---- pull-gather tables: for every peer s, the rows of s's chunk I need and where they go in my compact buffer ----
            owner = need // R
            src_idx = [(need[owner == s] - s * R).to(dev) for s in range(world)]
            offs = [int((owner < s).sum()) for s in range(world)]
            cnts = [int((owner == s).sum()) for s in range(world)]
            peers_x = [hx.get_buffer(s, (R, F), torch.bfloat16) for s in range(world)]

            src_idx = [t.to(torch.int32) for t in src_idx]

            def pull_gather(out):                                             # torch index_select with a remote source
                for s in range(world):
                    if cnts[s]:
                        torch.index_select(peers_x[s], 0, src_idx[s], out=out[offs[s]:offs[s] + cnts[s]])
                return out

            def pull_gather_t(out):                                           # Triton remote-read kernel per peer segment
                return pull_gather_triton(out, peers_x, src_idx, offs, cnts)

            src_rank_all = owner.to(torch.int32).to(dev)                      # per compact row: source rank and its row
            src_row_all = (need - owner * R).to(torch.int32).to(dev)

            def pull_gather_m(out):                                           # one kernel, peers interleaved
                return pull_gather_mixed(out, peers_x, src_rank_all, src_row_all)

            # ---- pull-reduce tables: for my own row i (block-local me*R + i): its position in peer s's compact buffer ----
            idx = torch.full((R, world), -1, dtype=torch.int32)
            lo, hi = rank * R, (rank + 1) * R
            for s in range(world):
                ns = needs[s]
                pos = torch.nonzero((ns >= lo) & (ns < hi)).squeeze(1)         # compact positions in s's buffer
                idx[ns[pos] - lo, s] = pos.to(torch.int32)
            idx = idx.to(dev)
            partial = symm_mem.empty((n_max, F), dtype=torch.bfloat16, device=dev)
            hp = symm_mem.rendezvous(partial, group=gname)
            partial.normal_()
            peers_p = [hp.get_buffer(s, (n_max, F), torch.bfloat16) for s in range(world)]
            out_g = torch.empty(n_need, F, dtype=torch.bfloat16, device=dev)
            out_r = torch.empty(R, F, dtype=torch.bfloat16, device=dev)
            # ---- correctness: pull-gather == plan.gather; pull-reduce == plan.reduce_scatter (fp32 reference) ----
            hx.barrier(channel=0)
            ref_g = plan.gather(x)
            pull_gather(out_g)
            err_g = (out_g.float() - ref_g.float()).abs().max().item()
            out_g.zero_(); pull_gather_t(out_g)
            err_g = max(err_g, (out_g.float() - ref_g.float()).abs().max().item())
            out_g.zero_(); pull_gather_m(out_g)
            err_g = max(err_g, (out_g.float() - ref_g.float()).abs().max().item())
            hp.barrier(channel=0)
            ref_r = plan.reduce_scatter(partial[:n_need].float())              # fp32 sum of the same payload
            pull_reduce(out_r, idx, peers_p)
            err_r = ((out_r.float() - ref_r).abs().max() / ref_r.abs().max().clamp_min(1e-6)).item()
            e = torch.tensor([err_g, err_r], device=dev); dist.all_reduce(e, op=dist.ReduceOp.MAX)
            # ---- timing ----
            y_full = torch.empty(world * R, F, dtype=torch.bfloat16, device=dev).normal_()
            t_ag = timeit_nccl(lambda: raw_all_gather_rows(x, group), args.iters)
            t_rs = timeit_nccl(lambda: raw_reduce_scatter_rows(y_full, group), args.iters)
            t_pg = timeit_nccl(lambda: plan.gather(x), args.iters)
            t_pr = timeit_nccl(lambda: plan.reduce_scatter(partial[:n_need]), args.iters)
            t_lg = timeit(lambda: pull_gather(out_g), args.iters, hx)
            t_lgt = timeit(lambda: pull_gather_t(out_g), args.iters, hx)
            t_lgm = timeit(lambda: pull_gather_m(out_g), args.iters, hx)
            t_lr = timeit(lambda: pull_reduce(out_r, idx, peers_p), args.iters, hp)
            mb_dense = (world - 1) * R * F * 2 / 1e6
            mb_sparse = (n_need - cnts[rank]) * F * 2 / 1e6
            say(f"PULL rows={R} dim={F} density={rho:.2f} dense_MB={mb_dense:.0f} sparse_MB={mb_sparse:.0f} "
                f"| gather: nccl_AG {t_ag:.2f} ms  a2av {t_pg:.2f}  pull_idxsel {t_lg:.2f}  pull_seq {t_lgt:.2f}  pull_mixed {t_lgm:.2f}  "
                f"| reduce: nccl_RS {t_rs:.2f}  a2av+spmm {t_pr:.2f}  pull_fused {t_lr:.2f}  "
                f"| check: gather maxabs {e[0].item():.1e} reduce rel {e[1].item():.1e}")
            del partial, peers_p, out_g, out_r, y_full, plan
            torch.cuda.synchronize()
        del x, peers_x
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
