"""Where does sparsity-aware communication (gt3d/dist/sparse_comm.py) pay?  The analytic table behind the selection rule.

Under the random (partition-free) id order a source row of the column block is needed by rank (r, c) when it has an edge
into row block r: with d_attn attention edges per node (incl. self loop, excl. expander) the share of the column block
referenced is 1 - exp(-d / P_d), and the 2 d_e expander edges add the ordering-independent floor 1 - (1 - 1/P_d)^(2 d_e)
(RESULTS.md 2026-10-10 / 2026-10-18).  The row side (Q gather, partials reduce-scatter) has the same share with P_o.
The sparse collective is an all-to-all-v, which runs per rank at a2a / AG = 0.89 x the all-gather rate when the group has
one rank per node, 0.40 x with two and 0.17 x with four ranks per node (NCCL's ring all-gather takes 3 of every 4 hops over
NVLink; bench/hw_constants_nccl_64bb.json), so the time ratio sparse / dense = share / (a2a / AG) and the plan is used
when it is below GT3D_SPARSE_MARGIN = 0.9 (sparse_comm.SparseRowPlan.worthwhile).

    python bench/sparse_comm_model.py [--gpus 64 256 1024] [--po 1 4 8 16]
"""
import argparse
import math

from gt3d.dist.sparse_comm import A2A_OVER_AG, MARGIN

GRAPHS = {"products": 57.5 - 6, "Reddit": 499.0 - 6, "papers100M": 36.2 - 6}     # d_attn without the 2 d_e expander edges
GPUS_PER_NODE = 4


def share(d_real, p, exp_deg=3):
    f_real = 1 - math.exp(-d_real / p)
    f_exp = 1 - (1 - 1 / p) ** (2 * exp_deg)
    return 1 - (1 - f_real) * (1 - f_exp)


def col_group_layout(pd, po, gpn=GPUS_PER_NODE):
    """(ranks of a column group per node, spans > 1 node) under the row-major rank map rank = r * P_o + c."""
    per_node = max(1, gpn // po)                      # ranks c, c + P_o, ... inside one node
    per_node = min(per_node, pd)
    multi = pd * po > gpn and pd > per_node
    return per_node, multi


def row_group_layout(pd, po, gpn=GPUS_PER_NODE):
    per_node = min(po, gpn)
    return per_node, po > gpn


def ratio(layout):
    per_node, multi = layout
    return A2A_OVER_AG.get((min(per_node, 4), multi), A2A_OVER_AG[(4, multi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--po", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--exp_deg", type=int, default=3)
    args = ap.parse_args()
    print(f"{'graph':11s} {'GPUs':>5s} {'grid':>8s} | {'col share':>9s} {'layout':>9s} {'a2a/AG':>6s} {'t_sp/t_de':>9s} {'col':>6s} | "
          f"{'row share':>9s} {'layout':>9s} {'a2a/AG':>6s} {'t_sp/t_de':>9s} {'row':>6s}")
    for name, d in GRAPHS.items():
        for P in args.gpus:
            for po in args.po:
                pd = P // po
                if pd < 1 or pd * po != P:
                    continue
                cells = []
                for sh, lay in ((share(d, pd, args.exp_deg), col_group_layout(pd, po)), (share(d, po, args.exp_deg), row_group_layout(pd, po))):
                    rt = ratio(lay)
                    t = sh / rt
                    cells.append(f"{sh:9.3f} {lay[0]:3d}/node{'*' if lay[1] else ' '} {rt:6.2f} {t:9.2f} {'SPARSE' if t <= MARGIN else 'dense':>6s}")
                print(f"{name:11s} {P:5d} {pd:4d}x{po:<3d} | {cells[0]} | {cells[1]}")
    print("\n* = the group spans several nodes; t_sp/t_de = share / (a2a/AG): the sparse collective's time relative to the dense one; "
          f"SPARSE when <= {MARGIN} (GT3D_SPARSE_MARGIN).")


if __name__ == "__main__":
    main()
