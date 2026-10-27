"""Which kernel should sum the rows a sparse reduce-scatter receives (gt3d/dist/sparse_comm.SparseRowPlan._accumulate)?
[n_send, F] received rows -> [n_rows, F] sums over the holders of every owned row, with a static structure
(send_idx: owner row of every received row, grouped by sender).  Candidates: permutation + torch_scatter.segment_csr,
index_add_ (atomics), a ones-valued CSR [n_rows, n_send] times the rows (cuSPERSE SpMM, fp32 accumulation), and, for
the one-holder case, index_copy_.  Single GPU:  python bench/bench_sparse_accumulate.py"""
import time

import torch
from torch_scatter import segment_csr


def t(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    dev = torch.device("cuda")
    for n_rows, holders, F, dt in [(612352, 1, 260, torch.bfloat16), (612352, 2, 256, torch.bfloat16), (100000, 38, 256, torch.bfloat16),
                                   (612352, 1, 260, torch.float32), (100000, 38, 256, torch.float32)]:
        n_send = n_rows * holders
        # holders senders, each holding every row (the worst case for a given share): send_idx = [arange] * holders
        send_idx = torch.arange(n_rows, device=dev).repeat(holders)
        recv = torch.randn(n_send, F, device=dev).to(dt)
        perm = torch.argsort(send_idx, stable=True)
        indptr = torch.zeros(n_rows + 1, dtype=torch.long, device=dev)
        indptr[1:] = torch.cumsum(torch.bincount(send_idx, minlength=n_rows), 0)
        mb = n_send * F * recv.element_size() / 1e6
        res = {}
        res["perm+segment_csr"] = t(lambda: segment_csr(recv.index_select(0, perm), indptr, reduce="sum"))
        res["perm+segment_csr fp32"] = t(lambda: segment_csr(recv.index_select(0, perm).float(), indptr, reduce="sum").to(dt))
        res["index_add_ (same dt)"] = t(lambda: torch.zeros(n_rows, F, dtype=dt, device=dev).index_add_(0, send_idx, recv))
        res["index_add_ fp32"] = t(lambda: torch.zeros(n_rows, F, dtype=torch.float32, device=dev).index_add_(0, send_idx, recv.float()).to(dt))
        ones = torch.ones(n_send, dtype=dt, device=dev)
        S = torch.sparse_csr_tensor(indptr.to(torch.int32), perm.to(torch.int32), ones, size=(n_rows, n_send))
        try:
            res["csr spmm (same dt)"] = t(lambda: torch.sparse.mm(S, recv))
        except Exception as e:  # noqa: BLE001
            res["csr spmm (same dt)"] = float("nan"); print("spmm", dt, "failed:", str(e)[:80])
        S32 = torch.sparse_csr_tensor(indptr.to(torch.int32), perm.to(torch.int32), ones.float(), size=(n_rows, n_send))
        res["csr spmm fp32"] = t(lambda: torch.sparse.mm(S32, recv.float()).to(dt))
        if holders == 1:
            res["index_copy_"] = t(lambda: torch.zeros(n_rows, F, dtype=dt, device=dev).index_copy_(0, send_idx, recv))
        res["copy (floor)"] = t(lambda: recv.clone())
        print(f"n_rows {n_rows} holders {holders} F {F} {str(dt).split('.')[-1]} payload {mb:.0f} MB: " +
              "  ".join(f"{k} {v:.2f} ms" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
