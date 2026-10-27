"""Triton prototype (Backend B) vs Backend A in fp32 on random graphs with empty rows
and one hub row; all score/norm mode combinations; forward + gradients."""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.attention.backend_torch import build_csr, masked_sparse_attention_torch  # noqa: E402
from gt3d.attention.triton_kernel import masked_sparse_attention_triton  # noqa: E402


def _case(n, e, h, d, t, seed, device):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = ei[:, ei[1] % 11 != 5]                                   # some empty destination rows
    hub = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)])  # row 0 attends to everyone
    ei = torch.cat([ei, hub], dim=1)
    etype = torch.randint(0, t, (ei.shape[1],), generator=g).to(torch.int32)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)
    return ei.to(device), etype.to(device), mk(n, h, d), mk(n, h, d), mk(n, h, d), mk(t, h, d), mk(t, h)


def run(score, norm, n, e, h, d, t, device, seed=0, seg_edges=512):
    ei, etype, q, k, v, gate, bias = _case(n, e, h, d, t, seed, device)
    crow, col, et, _ = build_csr(ei, n, etype)
    a_in = [x.clone().requires_grad_(True) for x in (q, k, v, gate, bias)]
    b_in = [x.clone().requires_grad_(True) for x in (q, k, v, gate, bias)]
    o_a = masked_sparse_attention_torch(a_in[0], a_in[1], a_in[2], crow, col, et, gate=a_in[3], bias=a_in[4], score=score, norm=norm)
    o_b = masked_sparse_attention_triton(b_in[0], b_in[1], b_in[2], crow, col, et, gate=b_in[3], bias=b_in[4], score=score, norm=norm, seg_edges=seg_edges)
    w = torch.randn_like(o_a)
    (o_a * w).sum().backward()
    (o_b * w).sum().backward()
    rel = lambda x, y: ((x - y).abs().max() / y.abs().max().clamp_min(1e-12)).item()
    errs = {"o": rel(o_b, o_a)}
    for name, a, b in zip(["q", "k", "v", "gate", "bias"], a_in, b_in):
        if a.grad is not None:
            errs[name] = rel(b.grad if b.grad is not None else torch.zeros_like(a), a.grad)
    return errs


def test_all():
    device = "cuda"
    worst = 0.0
    for (score, norm), (h, d), seg in itertools.product(
            itertools.product(["dot", "gated", "biased"], ["softmax", "clampexp"]),
            [(2, 40), (3, 8), (8, 32)], [512, 16]):
        errs = run(score, norm, n=1500, e=30000, h=h, d=d, t=3, device=device, seg_edges=seg)
        worst = max(worst, *errs.values())
        print(f"[triton] score={score:6s} norm={norm:8s} H={h} D={d:2d} seg={seg:3d}  " +
              "  ".join(f"{k}={v:.1e}" for k, v in errs.items()), flush=True)
        assert all(v < 2e-4 for v in errs.values()), (score, norm, h, d, seg, errs)
    print(f"[triton] ALL PASSED (worst rel err {worst:.2e})")


if __name__ == "__main__":
    test_all()
