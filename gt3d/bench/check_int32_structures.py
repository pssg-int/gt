import torch, time
dev = torch.device("cuda")
n_rows, n_cols, E = 200000, 150000, 3000000
src = torch.randint(0, n_cols, (E,), device=dev); dst = torch.randint(0, n_rows, (E,), device=dev)
perm = torch.argsort(dst * n_cols + src); src, dst = src[perm], dst[perm]
crow = torch.zeros(n_rows + 1, dtype=torch.long, device=dev); crow[1:] = torch.cumsum(torch.bincount(dst, minlength=n_rows), 0)
w = torch.rand(E, device=dev)
x = torch.randn(n_cols, 128, device=dev)
a64 = torch.sparse_csr_tensor(crow, src, w, size=(n_rows, n_cols))
a32 = torch.sparse_csr_tensor(crow.to(torch.int32), src.to(torch.int32), w, size=(n_rows, n_cols))
y64 = torch.sparse.mm(a64, x); y32 = torch.sparse.mm(a32, x)
print("int32 CSR spmm max abs diff:", (y64 - y32).abs().max().item(), "crow dtype", a32.crow_indices().dtype)
for name, a in (("int64", a64), ("int32", a32)):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): torch.sparse.mm(a, x)
    torch.cuda.synchronize(); print(f"spmm {name}: {(time.perf_counter()-t)/20*1e3:.2f} ms")
# transpose structure via argsort (no COO coalesce)
rows = torch.repeat_interleave(torch.arange(n_rows, device=dev), (crow[1:] - crow[:-1]).to(torch.int32))
print("repeat_interleave with int32 repeats OK:", rows.numel() == E)
pt = torch.argsort(src, stable=True); ccol = torch.zeros(n_cols + 1, dtype=torch.long, device=dev); ccol[1:] = torch.cumsum(torch.bincount(src, minlength=n_cols), 0)
at32 = torch.sparse_csr_tensor(ccol.to(torch.int32), rows[pt].to(torch.int32), w[pt], size=(n_cols, n_rows))
at_ref = torch.sparse_coo_tensor(torch.stack([src, rows]), w, size=(n_cols, n_rows)).coalesce().to_sparse_csr()
z = torch.randn(n_rows, 128, device=dev)
print("transpose via argsort vs coalesce max abs diff:", (torch.sparse.mm(at32, z) - torch.sparse.mm(at_ref, z)).abs().max().item())
# int32 crow through the Triton kernels (per-row and tiled)
from gt3d.attention.triton_kernel import attn_fwd_partial, attn_bwd_partial
h, d = 4, 32
q = torch.randn(n_rows, h, d, device=dev); k = torch.randn(n_cols, h, d, device=dev); v = torch.randn(n_cols, h, d, device=dev)
et = torch.randint(0, 2, (E,), device=dev, dtype=torch.int32); gate = torch.randn(2, h, d, device=dev)
col32 = src.to(torch.int32)
for thr in ("0", "100"):
    import os; os.environ["GT3D_TILED_THRESHOLD"] = thr
    import importlib, gt3d.attention.triton_kernel as tk; importlib.reload(tk)
    outs = []
    for cr in (crow, crow.to(torch.int32)):
        acc, l, m = tk.attn_fwd_partial(q, k, v, cr, col32, et, gate, None, score="gated", norm="clampexp")
        dq, dk, dv, dg, _ = tk.attn_bwd_partial(q, k, v, cr, col32, et, gate, None, m, torch.randn_like(acc).mul_(0) + 1.0, torch.ones_like(l), score="gated", norm="clampexp")
        outs.append((acc, l, dq, dk, dv, dg))
    diffs = [(a - b).abs().max().item() for a, b in zip(outs[0], outs[1])]
    print(f"tiled={'yes' if thr=='100' else 'no'} int64 vs int32 crow max diffs (acc,l,dq,dk,dv,dgate):", ["%.1e" % x for x in diffs])
