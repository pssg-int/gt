"""Re-implementation of DAC'26 (Lin, Madduri, Kandemir, "Scalable and Adaptive Parallel Training of Graph Transformer
on Large Graphs", arXiv 2604.16715; the authors' code is not public) on the paper's own stack: PyTorch 2.3.0+cu121 and
DGL 2.3.0 sparse operators, fp32.

Sparse Graph Attention (SGA), paper Eq. 3-5:  Q, K, V = X W_Q, X W_K, X W_V;  Z = (Q K^T) . A (SDDMM);
U = softmax(Z / sqrt(d')) (row-wise over the neighbours);  Y = U V (SpMM);  X' = X W_o + Y.
Multi-head attention splits the features into h subspaces (DGL's batched bsddmm / bspmm, the sparse Graph Transformer
example of DGL: q, k, v reshaped to [N, d', h]).

Strategies (rows = nodes, contiguous ranges of node ids per rank, the graph rows padded to a multiple of p):
  single  one GPU, no communication.
  gp_ag   paper Alg. 1: every rank projects Q/K/V of its N/p rows; all-gather K before the SDDMM and V before the
          SpMM (2 all-gathers forward, the matching 2 reduce-scatters backward); attention over the local rows (N/p x N
          block of A) for all h heads.
  gp_a2a  paper Alg. 2: every rank projects Q/K/V of its N/p rows; all-to-all of Q, K and V from node to head
          partitioning ([N/p, d', h] -> [N, d', h/p]); attention over the whole graph for h/p heads; all-to-all of the
          output back (4 all-to-alls forward, 4 backward); needs p | h.
The weights are replicated; their gradients are all-reduced once per iteration (one flat buffer).

    python dac26_sga.py --graph products --strategy gp_ag ...    (one process per GPU; RANK / WORLD_SIZE from srun)
    python dac26_sga.py --graph products --micro                 (paper Table 2: MM / SpMM / SDDMM on one GPU)

Prints one RESULT line per run.
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import dgl.sparse as dglsp

GRAPHS = "/pscratch/sd/c/cunyang/gnn/gt/baseline/dac26_graphs"
CLASSES = {"proteins": 112, "products": 47, "reddit": 41}


# ----------------------------------------------------------------------------------------------- collectives (autograd)
class AllGatherRows(torch.autograd.Function):
    """[n, ...] per rank -> [p n, ...] (rank order); backward: reduce-scatter of the gradient."""

    @staticmethod
    def forward(ctx, x):
        p = dist.get_world_size()
        out = torch.empty((p * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, g):
        p = dist.get_world_size()
        out = torch.empty((g.shape[0] // p,) + tuple(g.shape[1:]), dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(out, g.contiguous())
        return out


def _rows_to_heads(x, p):
    """[n, d', h] (my rows, all heads) -> [p n, d', h/p] (all rows, my head group); one all-to-all."""
    n, dh, h = x.shape
    send = x.reshape(n, dh, p, h // p).permute(2, 0, 1, 3).contiguous()      # block g = head group g -> rank g
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)                                        # recv[g] = rows of rank g, my heads
    return recv.reshape(p * n, dh, h // p)


def _heads_to_rows(y, p):
    """[p n, d', h/p] (all rows, my head group) -> [n, d', h] (my rows, all heads); one all-to-all."""
    pn, dh, hg = y.shape
    n = pn // p
    send = y.reshape(p, n, dh, hg).contiguous()                               # block g = rows of rank g -> rank g
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)                                        # recv[g] = my rows, head group g
    return recv.permute(1, 2, 0, 3).reshape(n, dh, p * hg)


class RowsToHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return _rows_to_heads(x, p)

    @staticmethod
    def backward(ctx, g):
        return _heads_to_rows(g, ctx.p), None


class HeadsToRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, p):
        ctx.p = p
        return _heads_to_rows(y, p)

    @staticmethod
    def backward(ctx, g):
        return _rows_to_heads(g, ctx.p), None


# ------------------------------------------------------------------------------------------------------------- model
class SGA(torch.nn.Module):
    """One Sparse Graph Attention layer (paper Eq. 3-5) under the given strategy."""

    def __init__(self, d, h, strategy, p):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.dh, self.strategy, self.p = d, h, d // h, strategy, p
        self.wq, self.wk, self.wv, self.wo = (torch.nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, A, x):
        n = x.shape[0]
        q = self.wq(x).view(n, self.dh, self.h) * self.dh ** -0.5
        k = self.wk(x).view(n, self.dh, self.h)
        v = self.wv(x).view(n, self.dh, self.h)
        if self.strategy == "gp_ag":                                           # Alg. 1
            k = AllGatherRows.apply(k)
            u = dglsp.bsddmm(A, q, k.transpose(0, 1)).softmax()
            v = AllGatherRows.apply(v)
            y = dglsp.bspmm(u, v)
        elif self.strategy == "gp_a2a":                                        # Alg. 2
            q = RowsToHeads.apply(q, self.p)
            k = RowsToHeads.apply(k, self.p)
            u = dglsp.bsddmm(A, q, k.transpose(0, 1)).softmax()
            v = RowsToHeads.apply(v, self.p)
            y = HeadsToRows.apply(dglsp.bspmm(u, v), self.p)
        else:
            u = dglsp.bsddmm(A, q, k.transpose(0, 1)).softmax()
            y = dglsp.bspmm(u, v)
        return self.wo(x) + y.reshape(n, self.d)


class Net(torch.nn.Module):
    def __init__(self, d, h, layers, classes, strategy, p):
        super().__init__()
        self.layers = torch.nn.ModuleList(SGA(d, h, strategy, p) for _ in range(layers))
        self.out = torch.nn.Linear(d, classes)

    def forward(self, A, x):
        for i, layer in enumerate(self.layers):
            x = layer(A, x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return self.out(x)


# -------------------------------------------------------------------------------------------------------------- data
def load_graph(name):
    g = torch.load(os.path.join(GRAPHS, f"{name}.pt"))
    return g["num_nodes"], g["edge_index"]


def local_matrix(n, ei, strategy, rank, p, dev):
    """Sparse attention mask of this rank: rows = destinations (queries), columns = sources (keys / values)."""
    n_pad = math.ceil(n / p) * p
    chunk = n_pad // p
    src, dst = ei[0], ei[1]
    if strategy == "gp_ag":                                                    # my N/p rows x all N columns
        lo, hi = rank * chunk, (rank + 1) * chunk
        m = (dst >= lo) & (dst < hi)
        idx = torch.stack([dst[m] - lo, src[m]])
        shape = (chunk, n_pad)
    else:                                                                      # whole graph (single / gp_a2a)
        idx = torch.stack([dst, src])
        shape = (n_pad, n_pad)
    return dglsp.spmatrix(idx.to(dev), shape=shape), n_pad, chunk, int(idx.shape[1])


# ------------------------------------------------------------------------------------------------------------ modes
def cuda_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def micro(args, dev):
    """Paper Table 2: MM (N x d x d), SpMM (N x N x d), SDDMM (N x d x N) on one GPU; single-head (d = 128) and the
    batched multi-head form used by the model (h heads of d / h)."""
    n, ei = load_graph(args.graph)
    A = dglsp.spmatrix(torch.stack([ei[1], ei[0]]).to(dev), shape=(n, n))
    d, h = args.hidden, args.heads
    x = torch.randn(n, d, device=dev)
    w = torch.randn(d, d, device=dev)
    res = {"MM": cuda_ms(lambda: x @ w),
           "SpMM": cuda_ms(lambda: dglsp.spmm(A, x)),
           "SDDMM": cuda_ms(lambda: dglsp.sddmm(A, x, x.t()))}
    q = torch.randn(n, d // h, h, device=dev)
    Ah = dglsp.val_like(A, torch.rand(A.nnz, h, device=dev))
    res["bSpMM"] = cuda_ms(lambda: dglsp.bspmm(Ah, q))
    res["bSDDMM"] = cuda_ms(lambda: dglsp.bsddmm(A, q, q.transpose(0, 1)))
    res["softmax"] = cuda_ms(lambda: Ah.softmax())
    print(f"RESULT micro graph={args.graph} N={n} E={ei.shape[1]} d={d} heads={h} tf32={int(torch.backends.cuda.matmul.allow_tf32)} "
          + " ".join(f"{k}_ms={v:.2f}" for k, v in res.items()), flush=True)


def kernel(args, dev):
    """One SGA attention (SDDMM -> softmax -> SpMM, h heads) forward + backward on the whole graph, one GPU: the
    operator of the paper's stack, for the like-for-like comparison with ours (gt3d/bench/bench_sga_kernel.py)."""
    n, ei = load_graph(args.graph)
    A = dglsp.spmatrix(torch.stack([ei[1], ei[0]]).to(dev), shape=(n, n))
    dh, h = args.hidden // args.heads, args.heads
    g = torch.Generator(device="cpu").manual_seed(0)                          # drawn as [N, h, d'] like ours
    q, k, v = (torch.randn(n, h, dh, generator=g).permute(0, 2, 1).contiguous().to(dev).requires_grad_()
               for _ in range(3))
    dout = torch.randn(n, h, dh, generator=g).permute(0, 2, 1).contiguous().to(dev)

    def fwd():
        return dglsp.bspmm(dglsp.bsddmm(A, q * dh ** -0.5, k.transpose(0, 1)).softmax(), v)

    if args.dump:                                                              # reference for bench_sga_kernel.py --check
        out = fwd(); out.backward(dout)
        t = lambda x: x.detach().permute(0, 2, 1).contiguous().cpu()          # noqa: E731  [N, d', h] -> [N, h, d']
        torch.save({"out": t(out), "dq": t(q.grad), "dk": t(k.grad), "dv": t(v.grad)}, args.dump)
        q.grad = k.grad = v.grad = None
    for _ in range(3):
        fwd().backward(dout)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    s, m, e = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    f_ms = b_ms = 0.0
    for _ in range(args.iters):
        s.record(); out = fwd(); m.record(); out.backward(dout); e.record()
        torch.cuda.synchronize()
        f_ms += s.elapsed_time(m); b_ms += m.elapsed_time(e)
        del out
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    print(f"RESULT kernel impl=dgl graph={args.graph} N={n} E={ei.shape[1]} heads={h} head_dim={dh} "
          f"fwd_ms={f_ms / args.iters:.2f} bwd_ms={b_ms / args.iters:.2f} peak_extra_gb={peak:.2f}", flush=True)


def train(args, dev, rank, p):
    n, ei = load_graph(args.graph)
    strategy = args.strategy if p > 1 else "single"
    if strategy == "gp_a2a":
        assert args.heads % p == 0, "GP-A2A needs p | heads"
    A, n_pad, chunk, e_loc = local_matrix(n, ei, strategy, rank, p, dev)
    del ei
    torch.manual_seed(args.seed)                                               # identical weights on every rank
    model = Net(args.hidden, args.heads, args.layers, CLASSES[args.graph], strategy, p).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    x_all = torch.randn(n_pad, args.hidden, generator=g)                       # input features after the input encoder
    y_all = torch.randint(0, CLASSES[args.graph], (n_pad,), generator=g)
    lo = rank * chunk if strategy != "single" else 0
    rows = chunk if strategy != "single" else n_pad
    x = x_all[lo:lo + rows].to(dev)
    y = y_all[lo:lo + rows].to(dev)
    valid = (torch.arange(lo, lo + rows) < n).to(dev)                          # padding rows carry no loss
    del x_all, y_all
    params = [q for q in model.parameters()]

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(A, x)
        loss = F.cross_entropy(logits[valid], y[valid], reduction="sum") / n
        loss.backward()
        if p > 1:                                                              # replicated weights: one flat all-reduce
            flat = torch.cat([q.grad.reshape(-1) for q in params])
            dist.all_reduce(flat)
            off = 0
            for q in params:
                q.grad.copy_(flat[off:off + q.numel()].view_as(q.grad)); off += q.numel()
        opt.step()

    times = []
    torch.cuda.reset_peak_memory_stats()
    for it in range(args.warmup + args.iters):
        if p > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        if it >= args.warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor([sum(times) / len(times), torch.cuda.max_memory_allocated() / 2 ** 30, float(e_loc)], device=dev)
    if p > 1:
        tmax = t.clone(); dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
        tmin = t.clone(); dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
    else:
        tmax = tmin = t
    if rank == 0:
        print(f"RESULT train graph={args.graph} strategy={strategy} gpus={p} N={n} hidden={args.hidden} heads={args.heads} "
              f"layers={args.layers} ms_per_iter={tmax[0].item():.1f} peak_gb={tmax[1].item():.2f} "
              f"edges_per_rank_max={int(tmax[2].item())} edges_per_rank_min={int(tmin[2].item())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, choices=list(CLASSES))
    ap.add_argument("--strategy", default="gp_ag", choices=["gp_ag", "gp_a2a"])
    ap.add_argument("--hidden", type=int, default=128)                         # paper 5.1: hidden 128, 8 heads
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)                           # paper 5.4: 3-layer model
    ap.add_argument("--iters", type=int, default=10)                           # paper 5.1: mean of 10 runs after 2 warm-up
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", action="store_true", help="paper Table 2 kernel microbenchmark (one GPU)")
    ap.add_argument("--kernel", action="store_true", help="one SGA attention fwd + bwd on the whole graph (one GPU)")
    ap.add_argument("--dump", default=None, help="(--kernel) save output and q/k/v gradients of one call to this file")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False                              # fp32 (PyTorch 2.3 default)
    p = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dev = torch.device("cuda")
    if args.micro or args.kernel:
        (micro if args.micro else kernel)(args, dev)
        return
    if p > 1:
        dist.init_process_group("nccl")
    train(args, dev, rank, p)
    if p > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
