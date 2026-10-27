"""Feistel permutations are bijections; block-wise expander generation equals the global one."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gt3d.data.expander import FeistelPermutation, feistel_expander_edges  # noqa: E402


def test_bijection():
    for n in [1, 2, 3, 7, 1000, 12345, 1 << 17, (1 << 17) + 5]:
        pi = FeistelPermutation(n, seed=3)
        y = pi(torch.arange(n))
        assert y.min() >= 0 and y.max() < n and torch.unique(y).numel() == n, n
    print("[expander] bijection OK")


def test_block_equals_global():
    import torch.distributed as dist
    from gt3d.dist.grid2d import Grid2D
    from gt3d.data.expander import block_expander_edges
    # emulate a grid without initialising torch.distributed: patch the pieces Grid2D needs
    class FakeGrid(Grid2D):
        def __init__(self, p_rows, p_cols, num_nodes, rank):
            self.p_rows, self.p_cols, self.rank = p_rows, p_cols, rank
            self.r, self.c = divmod(rank, p_cols)
            self.num_nodes = num_nodes
            self.chunk = -(-num_nodes // (p_rows * p_cols)); self.n_pad = self.chunk * p_rows * p_cols
    n, deg = 5003, 3
    glob = feistel_expander_edges(n, deg, seed=11)
    for p_rows, p_cols in [(2, 2), (3, 2), (1, 4)]:
        total = 0
        for rank in range(p_rows * p_cols):
            g = FakeGrid(p_rows, p_cols, n, rank)
            blk = block_expander_edges(g, n, deg, seed=11)
            ref, _ = g.local_block(glob)
            a = set(map(tuple, blk.t().tolist())); b = set(map(tuple, ref.t().tolist()))
            assert a == b, (p_rows, p_cols, rank, len(a), len(b), len(a ^ b))
            total += len(a)
        n_unique = torch.unique(glob, dim=1).shape[1]             # the multigraph may contain duplicate edges
        assert total == n_unique, (total, n_unique)
    print(f"[expander] block generation == global on 3 grids ({glob.shape[1]} edges) OK")


if __name__ == "__main__":
    test_bijection()
    test_block_equals_global()
