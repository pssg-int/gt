"""Backend A (CSR / torch ops) vs the edge-list reference, fp64, all mode combinations,
including chunked execution and rows without in-edges."""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.ref import masked_sparse_attention_ref  # noqa: E402
from gt3d.attention.backend_torch import build_csr, masked_sparse_attention_torch  # noqa: E402


def _case(n=97, e=700, h=3, d=8, t=3, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = ei[:, ei[1] % 7 != 3]          # leave some destination rows empty
    etype = torch.randint(0, t, (ei.shape[1],), generator=g)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(device)
    q, k, v = mk(n, h, d), mk(n, h, d), mk(n, h, d)
    gate, bias = mk(t, h, d), mk(t, h)
    return ei.to(device), etype.to(device), q, k, v, gate, bias


def run_one(score, norm, chunk_edges, device):
    ei, etype, q, k, v, gate, bias = _case(device=device)
    n = q.shape[0]
    params = [q, k, v, gate, bias]
    ref_in = [p.clone().requires_grad_(True) for p in params]
    tst_in = [p.clone().requires_grad_(True) for p in params]

    o_ref = masked_sparse_attention_ref(ref_in[0], ref_in[1], ref_in[2], ei, edge_type=etype,
                                        gate=ref_in[3], bias=ref_in[4], score=score, norm=norm)
    crow, col, et_sorted, _ = build_csr(ei, n, etype)
    o_tst = masked_sparse_attention_torch(tst_in[0], tst_in[1], tst_in[2], crow, col, et_sorted,
                                          gate=tst_in[3], bias=tst_in[4], score=score, norm=norm,
                                          chunk_edges=chunk_edges)
    err_f = (o_ref - o_tst).abs().max().item()
    w = torch.randn_like(o_ref)
    (o_ref * w).sum().backward()
    (o_tst * w).sum().backward()
    errs = {}
    for name, a, b in zip(["q", "k", "v", "gate", "bias"], ref_in, tst_in):
        if a.grad is None and b.grad is None:
            continue
        ga = a.grad if a.grad is not None else torch.zeros_like(a)
        gb = b.grad if b.grad is not None else torch.zeros_like(b)
        errs[name] = (ga - gb).abs().max().item()
    return err_f, errs


def test_all_modes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    worst = 0.0
    for score, norm, chunk in itertools.product(["dot", "gated", "biased"], ["softmax", "clampexp"], [0, 50]):
        err_f, errs = run_one(score, norm, chunk, device)
        worst = max(worst, err_f, *errs.values())
        print(f"[backend_torch] score={score:6s} norm={norm:8s} chunk={chunk:3d} "
              f"fwd={err_f:.2e} grads={{{', '.join(f'{k}={v:.1e}' for k, v in errs.items())}}}")
        assert err_f < 1e-12 and all(e < 1e-11 for e in errs.values()), (score, norm, chunk, err_f, errs)
    print(f"[backend_torch] ALL PASSED (worst abs err {worst:.2e}, device={device})")


if __name__ == "__main__":
    test_all_modes()
