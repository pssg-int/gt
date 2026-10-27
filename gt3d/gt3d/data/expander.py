"""Expander-edge generation.

Follows the official Exphormer 'Random-d' construction (union of `degree` random
permutations, symmetrized, self loops removed) but without the Laplacian
lambda_2 selection loop, which is infeasible beyond ~1e6 nodes.  Random d-regular
graphs are expanders with high probability (Friedman's theorem), which is all
the Exphormer analysis needs.  A distributed, hash-based variant (per-block
generation without communication) will live next to this function.
"""
import torch


def random_regular_expander_edges(num_nodes: int, degree: int, seed: int = 0,
                                  device="cpu") -> torch.Tensor:
    """Return edge_index [2, E] (src, dst) of a random 2*degree-regular multigraph
    minus self loops, generated from `degree` random permutations."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    senders = torch.arange(num_nodes).repeat(degree)
    receivers = torch.cat([torch.randperm(num_nodes, generator=g) for _ in range(degree)])
    src = torch.cat([senders, receivers])
    dst = torch.cat([receivers, senders])
    keep = src != dst
    return torch.stack([src[keep], dst[keep]]).to(device)


# ---------------------------------------------------------------------------------------------
# Hash-based (Feistel) pseudo-random permutations: every rank can enumerate exactly the expander
# edges that fall into its own 2D block without generating or communicating the global edge list.
# ---------------------------------------------------------------------------------------------
_MASK32 = (1 << 32) - 1


def _mix(x: torch.Tensor, key: int) -> torch.Tensor:
    """32-bit integer hash of x with a round key (murmur3-style finaliser), int64 arithmetic."""
    h = (x ^ key) & _MASK32
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & _MASK32
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & _MASK32
    return h ^ (h >> 16)


class FeistelPermutation:
    """Bijection on [0, n) built from a 4-round balanced Feistel network on 2*k bits with
    cycle walking (values >= n are re-encrypted until they land below n).  Deterministic in
    (n, seed, index); evaluates elementwise on int64 tensors."""

    def __init__(self, n: int, seed: int, index: int = 0, rounds: int = 4):
        self.n = int(n)
        bits = max(2, (self.n - 1).bit_length())
        self.k = (bits + 1) // 2                      # half-width
        self.half_mask = (1 << self.k) - 1
        self.domain = 1 << (2 * self.k)
        g = torch.Generator().manual_seed(seed * 1_000_003 + index)
        self.keys = [int(v) for v in torch.randint(0, 1 << 31, (rounds,), generator=g)]

    def _encrypt_once(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x >> self.k, x & self.half_mask
        for key in self.keys:
            left, right = right, left ^ (_mix(right, key) & self.half_mask)
        return (left << self.k) | right

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self._encrypt_once(x.to(torch.int64))
        bad = y >= self.n
        while bool(bad.any()):                          # cycle walking; expected < 4 iterations
            y = torch.where(bad, self._encrypt_once(y), y)
            bad = y >= self.n
        return y


def feistel_expander_edges(num_nodes: int, degree: int, seed: int = 0, device="cpu") -> torch.Tensor:
    """Global edge list [2, E] (src, dst) of the union of `degree` Feistel permutations, both
    directions, self loops removed.  Same construction as random_regular_expander_edges but with
    permutations that any rank can evaluate pointwise (see block_expander_edges)."""
    ar = torch.arange(num_nodes, device=device)
    srcs, dsts = [], []
    for k in range(degree):
        pi = FeistelPermutation(num_nodes, seed, k)(ar)
        srcs += [ar, pi]
        dsts += [pi, ar]
    src, dst = torch.cat(srcs), torch.cat(dsts)
    keep = src != dst
    return torch.stack([src[keep], dst[keep]])


def block_expander_edges(grid, num_nodes: int, degree: int, seed: int = 0, device="cpu"):
    """The subset of feistel_expander_edges(num_nodes, degree, seed) whose dst lies in the row block
    and src in the column block of `grid`'s rank, computed from the block's own node ranges only
    (O((N/P_o + N/P_d) * degree) work, no communication).  Returns LOCAL (col-local src,
    row-local dst) indices, matching Grid2D.local_block."""
    n = num_nodes
    c_lo, c_hi = grid.col_block_range()
    col_ids = torch.arange(c_lo, min(c_hi, n), device=device)                       # real nodes of my column block
    row_ids = torch.cat([torch.arange(lo, min(hi, n), device=device) for lo, hi in grid.row_block_ranges()])
    srcs, dsts = [], []
    for k in range(degree):
        pi = FeistelPermutation(n, seed, k)
        # edges (i, pi(i)) and (pi(i), i) for all i: enumerate the endpoint that lives in my column
        # block (src side) and the endpoint that lives in my row block (dst side).
        d = pi(col_ids)                                   # src = col_ids, dst = pi(col_ids)
        keep = (grid.owner_of(d)[0].to(device) == grid.r) & (d != col_ids)
        srcs.append(col_ids[keep]); dsts.append(d[keep])
        s = pi(row_ids)                                   # dst = row_ids, src = pi(row_ids)
        keep = (s >= c_lo) & (s < c_hi) & (s != row_ids)
        srcs.append(s[keep]); dsts.append(row_ids[keep])
    src, dst = torch.cat(srcs), torch.cat(dsts)
    return torch.stack([grid.global_to_col_local(src), grid.global_to_row_local(dst)])
