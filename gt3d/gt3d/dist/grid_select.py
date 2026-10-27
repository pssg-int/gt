"""White-box performance and memory model of the distributed layers, and the grid chooser built on it.

Nothing in this module is fitted to training runs.  Every term is a count (bytes moved, FLOPs, rows, edges, saved
tensors) multiplied by a hardware or kernel constant that was measured ONCE in isolation on this machine
(bench/hw_constants.py: HBM, elementwise, launch, GEMM, NCCL alpha-beta per group placement; bench/kernel_theory.py and
bench/kernel_l2.py: HBM read bandwidth, latency per chunk and L2 factor of the attention kernels, whose time is analytic
(KERNEL_MODEL = "theory"); bench/kernel_constants.py: SpMM and tiled-kernel costs; train_dist.py --saved_census: the tensors
autograd keeps for backward).  The model therefore explains WHY a grid is fast, and its residual against a measured
epoch is itself informative (synchronisation / launch skew the model does not contain).

Geometry (per rank; P = P_d P_o ranks, row-major layout, GPN GPUs per node):
    O = N / P        owned rows                 R = N / P_d   row block (queries)      C = N / P_o   column block (keys)
    E_a / P          attention edges            E_g / P       GCN edges
    row group  = P_o consecutive ranks (intra-node iff P_o <= GPN);  column group strides by P_o (intra iff P <= GPN).

Fused 2D layer, per rank and layer (b = wire bytes, bf16 = 2):
    communication   AG x_col (C-O) d b over the column group, AG x_row (R-O) d b over the row group,
                    RS partials (R-O)(2d+H) b over the row group; backward: the mirror image (AG / RS swapped)
    GEMM            fwd 2 d^2 (3C + R) + 8 O d^2 (FFN);  bwd 2x (grad_input + grad_W)
    attention       per program max(ceil(d/16) H_g tau, lambda bytes / B), summed over the rows (columns) of the block
    GCN SpMM        s_e E_g/P + s_r R + s_c C per direction (cuSPARSE CSR)
    elementwise     counted passes over the owned / gathered tensors at the measured elementwise rate + per-op floor
Head grid (P_d, P_o, P_i), attention operator: the owner projects once (3 GEMMs on O rows), Ulysses all-to-all inside the
head group, gathers of the d/P_i head slices, the kernel on H/P_i heads, combine, all-to-all back; backward mirrored.
predict_head_ms = L x (that operator + the owned-row tail) + fixed work: the attention-only model on the head grid.
Streamed layer: the column block in n_sub sub-blocks (gather overlapped with the previous sub-block's compute),
DCSR / compact kernels on the non-empty (row, sub-block) pairs, everything recomputed in backward.
Memory: saved tensors per layer counted by --saved_census (3 U_col + 2 U_row + 10.9 U_own fp32 + 1.3 U_own of dropout
masks, U_x = rows_x d 4 B) + structures + features + the backward transient of one layer.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

GB = 2 ** 30
GPUS_PER_NODE = 4

# ------------------------------------------------------------------------------------------------------------
# measured constants (Perlmutter, A100-SXM4-40GB, Slingshot-11, NCCL 2.24 OFI; bench/hw_constants.json and
# bench/kernel_constants.json, 2026-10-17, allocation 59904654)
# ------------------------------------------------------------------------------------------------------------
HW = dict(
    hbm_gbs=1377.0,                 # 1 GB copy
    ew_gbs=600.0,                   # elementwise rate on >= 150K-row tensors (LN / dropout / add segment, fwd + bwd)
    op_us=12.5,                     # per-op floor of a small-tensor elementwise op (launch + dispatcher)
    fwd_op_us=7.8, bwd_node_us=8.8, # CPU cost per forward op / per autograd node (chain of 300 small ops)
    # GEMM TFLOPS of a Linear on 612K rows: {precision: {d: (forward, grad_input, grad_W)}}
    gemm={"ieee": {64: (15.9, 16.1, 14.9), 80: (12.9, 13.0, 10.6), 128: (18.2, 18.3, 17.1), 256: (18.7, 18.8, 18.3)},
          "tf32": {64: (21.1, 21.2, 21.7), 80: (25.9, 25.9, 26.2), 128: (42.9, 42.8, 43.7), 256: (79.4, 78.1, 79.9)}},
    # NCCL alpha (ms, 0.25 MB message) / beta (GB/s per rank of received bytes, 160 MB) per group LAYOUT "k x m" = k ranks per node on m nodes
    # (bench/hw_constants.py HW_ONLY=nccl): the row / column / head groups of every grid have this form.  Within a node
    # the rate is NVLink's; across nodes the m-node ring is limited by the NICs, and k ranks per node share the node's
    # k NICs.  m > 4 uses the m = 4 entry (a ring's per-rank rate does not depend on its length).
    nccl={(4, 1): dict(ag=(0.04, 186.8), rs=(0.05, 200.0), a2a=(0.12, 172.8)),
          (2, 1): dict(ag=(0.04, 54.0), rs=(0.04, 60.4), a2a=(0.05, 62.8)),
          (1, 2): dict(ag=(0.19, 14.8), rs=(0.26, 16.4), a2a=(0.10, 15.5)),
          (1, 4): dict(ag=(0.13, 16.4), rs=(0.30, 16.7), a2a=(0.09, 15.2)),
          (2, 2): dict(ag=(0.11, 33.7), rs=(0.11, 33.5), a2a=(0.09, 24.9)),
          (2, 4): dict(ag=(0.52, 33.9), rs=(0.18, 34.0), a2a=(0.10, 18.3)),
          (4, 2): dict(ag=(0.18, 70.6), rs=(0.15, 69.3), a2a=(0.10, 24.4)),
          (4, 4): dict(ag=(0.83, 70.2), rs=(0.62, 69.7), a2a=(0.11, 13.9)),
          # 8- and 16-node layouts (16-node job 59950338, 2026-10-18, bench/hw_constants_nccl_64bb.json): the same rates as
          # the 4-node rows of the same ranks-per-node — a ring across 8 or 16 nodes loses nothing in isolation
          (1, 8): dict(ag=(0.26, 16.7), rs=(0.54, 16.7), a2a=(0.12, 15.2)),
          (2, 8): dict(ag=(0.65, 34.2), rs=(0.56, 34.3), a2a=(0.42, 13.9)),
          (4, 8): dict(ag=(1.30, 68.6), rs=(0.83, 69.0), a2a=(0.95, 11.8)),
          (1, 16): dict(ag=(0.50, 16.7), rs=(0.38, 16.7), a2a=(0.14, 14.8)),
          (2, 16): dict(ag=(0.87, 34.3), rs=(0.85, 34.3), a2a=(0.35, 13.6)),
          (4, 16): dict(ag=(2.16, 69.6), rs=(1.15, 69.6), a2a=(0.88, 11.6))},
    # attention kernels, per-row regime (>= 3.5 edges per row): ns per edge / per row / per column, {(H, d_h): ...}
    attn_rows={(2, 40): dict(fwd=(0.48, 2.6, 0.4), bwd=(1.70, 6.0, 6.6)),
               (4, 32): dict(fwd=(0.70, 1.6, 0.7), bwd=(1.61, 9.8, 6.1)),
               (8, 32): dict(fwd=(1.41, 4.0, 1.0), bwd=(3.28, 19.1, 12.2)),     # 4 heads per program (2026-10-17)
               (1, 32): dict(fwd=(0.17, 0.9, 0.6), bwd=(0.38, 2.8, 1.6)),
               (2, 32): dict(fwd=(0.35, 1.4, 0.6), bwd=(0.82, 4.8, 3.0)),
               (1, 40): dict(fwd=(0.27, 1.0, 0.7), bwd=(0.68, 3.9, 3.2)),
               (2, 20): dict(fwd=(0.27, 1.6, 0.5), bwd=(0.73, 4.9, 3.5))},
    # the same with bf16 Q / K / V / dacc operands (--kernel_bf16, opt-in): forward cheaper per edge, backward dearer per row / column
    attn_rows_bf16={(2, 40): dict(fwd=(0.33, 2.9, 0.2), bwd=(1.42, 10.5, 9.2)),
                    (4, 32): dict(fwd=(0.45, 4.3, 0.3), bwd=(1.66, 12.3, 10.1)),
                    (8, 32): dict(fwd=(0.90, 8.7, 0.4), bwd=(3.33, 24.6, 20.2))},
    # attention kernels, compact (DCSR tiled) regime of the streamed sub-blocks: ns per edge / per NON-EMPTY row / per column
    attn_compact={(2, 40): dict(fwd=(1.67, 3.0, 0.0), bwd=(4.01, 4.2, 2.0)),
                  (4, 32): dict(fwd=(1.80, 2.9, 0.0), bwd=(4.57, 4.6, 1.5)),
                  (8, 32): dict(fwd=(3.49, 5.4, 0.0), bwd=(10.03, 11.7, 5.3))},
    # cuSPARSE CSR SpMM (GCN): ns per edge / per row / per column, {d: (A x, A^T g)}
    spmm={32: ((0.16, 0.34, 0.37), (0.16, 0.36, 0.34)),       # head-grid feature slices d / P_i (kernel_constants_headslices, 10-17)
          40: ((0.19, 0.44, 0.47), (0.19, 0.47, 0.44)), 64: ((0.21, 0.63, 0.49), (0.21, 0.49, 0.63)),
          80: ((0.27, 0.8, 0.7), (0.27, 0.7, 0.8)), 128: ((0.37, 1.3, 0.8), (0.37, 0.8, 1.3)),
          256: ((0.75, 2.6, 1.6), (0.75, 1.6, 2.6))},
    # DCSR compact SpMM of the streamed sub-blocks (rows = non-empty rows): {d: (A x, A^T g)}
    spmm_compact={80: ((0.59, 0.7, 0.1), (0.31, 0.0, 0.6)), 128: ((0.90, 1.0, 0.0), (0.43, 0.0, 0.9)),
                  256: ((1.79, 2.0, 0.1), (0.88, 0.0, 1.7))},
    # memory: tensors autograd saves per layer (fused layer, no checkpoint; --saved_census 2026-10-17), in units of
    # U_x = rows_x * hidden * 4 B;  the owned-row count includes the pre-MP share (4-layer models)
    saved=dict(col=3.0, row=2.03, own_f32=10.9, own_mask=1.3,
               # head layer (attention-only, Grid3D; --saved_census 2026-10-18, products h128 (1,1,4)): owned-chunk fp32
               # tensors 8.98 U_own, dropout masks 1.06 U_own per layer; the gathered head slices and the combine output
               # are counted analytically in peak_memory_head_bytes
               head_own_f32=9.0, head_own_mask=1.06,
               # with the GCN branch (census 2026-10-18 evening, products h128 (1,1,4)): +1.0 U_own fp32, +0.25 masks per layer
               head_gcn_own_f32=1.0, head_gcn_own_mask=0.25,
               # ring layer (attention-only; --saved_census 2026-10-26, products h128 (4, 1) ring, 4 layers): 40.2 U_own fp32
               # and 4.25 U_own of masks over the model = 10.05 + 1.06 per layer (layer input, packed partial, tail)
               ring_own_f32=10.05, ring_own_mask=1.06),
)


# measured call times (ms) per layout (ranks per node, nodes) and per-rank chunk size (MB, bf16) of all-gather /
# reduce-scatter / all-to-all, 16-node job 59950338 (2026-10-18, NCCL 2.27.3, every group of a layout running concurrently);
# coll_ms interpolates these points in log(size) instead of fitting alpha + bytes / beta (the pair above is kept for the
# calls that do not know their group size).
NCCL_CURVES = {
    (2, 1): dict(ag=[(0.25, 0.085), (1, 0.052), (4, 0.117), (9, 0.222), (16, 0.376), (64, 1.345), (160, 3.100)],
              rs=[(0.25, 0.048), (1, 0.046), (4, 0.105), (9, 0.240), (16, 0.312), (64, 1.158), (160, 2.799)],
              a2a=[(0.25, 0.061), (1, 0.070), (4, 0.077), (9, 0.162), (16, 0.189), (64, 0.639), (160, 1.343)]),
    (4, 1): dict(ag=[(0.25, 0.076), (1, 0.062), (4, 0.121), (9, 0.210), (16, 0.333), (64, 1.144), (160, 2.711)],
              rs=[(0.25, 0.048), (1, 0.056), (4, 0.109), (9, 0.193), (16, 0.307), (64, 1.059), (160, 2.543)],
              a2a=[(0.25, 0.061), (1, 0.061), (4, 0.058), (9, 0.078), (16, 0.174), (64, 0.334), (160, 0.731)]),
    (1, 2): dict(ag=[(0.25, 0.381), (1, 0.132), (4, 0.390), (9, 0.791), (16, 1.259), (64, 4.628), (160, 11.481)],
              rs=[(0.25, 0.198), (1, 0.124), (4, 0.346), (9, 0.753), (16, 1.201), (64, 4.152), (160, 10.242)],
              a2a=[(0.25, 0.107), (1, 0.102), (4, 0.218), (9, 0.413), (16, 0.660), (64, 2.297), (160, 5.478)]),
    (2, 2): dict(ag=[(0.25, 0.139), (1, 0.170), (4, 0.780), (9, 1.128), (16, 1.728), (64, 6.229), (160, 14.958)],
              rs=[(0.25, 0.119), (1, 0.532), (4, 0.480), (9, 1.074), (16, 1.689), (64, 6.188), (160, 15.349)],
              a2a=[(0.25, 0.093), (1, 0.578), (4, 0.210), (9, 0.479), (16, 0.645), (64, 2.183), (160, 5.096)]),
    (4, 2): dict(ag=[(0.25, 0.191), (1, 0.213), (4, 0.495), (9, 1.125), (16, 1.914), (64, 7.315), (160, 17.292)],
              rs=[(0.25, 0.174), (1, 0.230), (4, 0.554), (9, 1.110), (16, 1.894), (64, 7.043), (160, 17.065)],
              a2a=[(0.25, 0.094), (1, 0.128), (4, 0.287), (9, 0.472), (16, 0.735), (64, 2.668), (160, 6.233)]),
    (1, 4): dict(ag=[(0.25, 0.771), (1, 0.276), (4, 0.893), (9, 2.064), (16, 3.228), (64, 12.429), (160, 30.695)],
              rs=[(0.25, 0.152), (1, 0.324), (4, 0.887), (9, 1.911), (16, 3.126), (64, 12.250), (160, 30.617)],
              a2a=[(0.25, 0.102), (1, 0.128), (4, 0.313), (9, 0.537), (16, 0.941), (64, 3.401), (160, 8.433)]),
    (2, 4): dict(ag=[(0.25, 1.009), (1, 0.455), (4, 1.082), (9, 2.390), (16, 3.872), (64, 14.246), (160, 35.036)],
              rs=[(0.25, 0.197), (1, 0.304), (4, 1.095), (9, 2.265), (16, 3.700), (64, 14.217), (160, 34.884)],
              a2a=[(0.25, 0.100), (1, 0.127), (4, 0.317), (9, 0.658), (16, 1.023), (64, 3.652), (160, 8.712)]),
    (4, 4): dict(ag=[(0.25, 0.982), (1, 0.608), (4, 1.104), (9, 2.268), (16, 4.006), (64, 15.326), (160, 37.488)],
              rs=[(0.25, 0.351), (1, 0.411), (4, 1.037), (9, 2.309), (16, 3.860), (64, 15.329), (160, 37.143)],
              a2a=[(0.25, 0.367), (1, 0.125), (4, 0.294), (9, 0.656), (16, 1.138), (64, 4.768), (160, 11.560)]),
    (1, 8): dict(ag=[(0.25, 0.258), (1, 0.537), (4, 1.888), (9, 4.251), (16, 7.222), (64, 28.251), (160, 70.285)],
              rs=[(0.25, 0.536), (1, 0.552), (4, 1.893), (9, 4.219), (16, 7.186), (64, 28.144), (160, 70.135)],
              a2a=[(0.25, 0.116), (1, 0.153), (4, 0.358), (9, 0.706), (16, 1.104), (64, 4.193), (160, 9.668)]),
    (2, 8): dict(ag=[(0.25, 0.650), (1, 1.480), (4, 2.126), (9, 4.620), (16, 7.728), (64, 29.628), (160, 73.644)],
              rs=[(0.25, 0.563), (1, 0.774), (4, 2.084), (9, 4.533), (16, 7.745), (64, 29.645), (160, 73.461)],
              a2a=[(0.25, 0.422), (1, 0.130), (4, 0.590), (9, 0.819), (16, 1.318), (64, 4.931), (160, 11.335)]),
    (4, 8): dict(ag=[(0.25, 1.301), (1, 1.072), (4, 2.073), (9, 4.467), (16, 8.097), (64, 30.597), (160, 75.806)],
              rs=[(0.25, 0.833), (1, 0.830), (4, 2.121), (9, 4.449), (16, 7.755), (64, 30.612), (160, 75.355)],
              a2a=[(0.25, 0.950), (1, 0.178), (4, 0.340), (9, 0.877), (16, 1.446), (64, 5.237), (160, 13.790)]),
    (1, 16): dict(ag=[(0.25, 0.503), (1, 1.124), (4, 3.934), (9, 8.743), (16, 15.291), (64, 60.811), (160, 150.670)],
              rs=[(0.25, 0.377), (1, 1.189), (4, 3.913), (9, 8.827), (16, 15.038), (64, 60.055), (160, 150.996)],
              a2a=[(0.25, 0.145), (1, 0.240), (4, 0.395), (9, 0.864), (16, 1.265), (64, 4.545), (160, 10.610)]),
    (2, 16): dict(ag=[(0.25, 0.874), (1, 1.827), (4, 4.157), (9, 9.083), (16, 15.806), (64, 60.576), (160, 151.836)],
              rs=[(0.25, 0.848), (1, 1.173), (4, 4.096), (9, 9.056), (16, 15.110), (64, 61.051), (160, 151.763)],
              a2a=[(0.25, 0.349), (1, 0.160), (4, 0.389), (9, 0.891), (16, 1.402), (64, 4.623), (160, 11.914)]),
    (4, 16): dict(ag=[(0.25, 2.161), (1, 2.357), (4, 4.070), (9, 9.093), (16, 15.621), (64, 61.242), (160, 151.939)],
              rs=[(0.25, 1.146), (1, 1.458), (4, 4.184), (9, 8.836), (16, 15.620), (64, 60.877), (160, 151.825)],
              a2a=[(0.25, 0.882), (1, 0.312), (4, 0.397), (9, 0.896), (16, 1.703), (64, 5.275), (160, 14.219)]),
}


@dataclass
class Graph:
    n: int                      # nodes
    e_attn: int                 # attention edges (graph + self loops + expander)
    e_gcn: int                  # GCN edges (graph + self loops)
    f_in: int = 128             # input features
    kappa: float = 0.8          # degree dispersion of the non-empty-pair model (negative binomial; 0.8 = papers100M)
    name: str = None            # per-graph kernel calibration table bench/graph_constants_<name>.json; None = random-block constants


# Per-graph kernel calibration (bench/graph_constants.py, one GPU, minutes, measured once per graph): the attention
# kernels (forward, backward) and the GCN SpMM (A x, A^T g) timed on the graph's REAL rank-0 blocks of every candidate
# grid shape (P_d, P_o) and head configuration (H, d_head).  With Graph(name=...) the model uses, for a block shape and
# configuration that was measured, the measured time itself instead of constants x counts; for other shapes the ratio
# measured / (random-block model) of the nearest measured K / V footprint scales the constants-x-counts term.  The
# random-block constants never see a real graph (clustered rows, degree skew, the 3-4 edges-per-row regime at the
# tiled threshold), which is where their 15-50 % shape errors come from; the per-graph table removes them where it
# matters and is NOT a fit to any training time.  Tables live in bench/graph_constants_<name>.json.
_GRAPH_TABLES = {}


def _graph_table(name):
    if name is None:
        return None
    if name not in _GRAPH_TABLES:
        import json, os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # gt3d/gt3d/dist -> gt3d/
        path = os.path.join(root, "bench", f"graph_constants_{name}.json")
        _GRAPH_TABLES[name] = json.load(open(path)) if os.path.exists(path) else None
    return _GRAPH_TABLES[name]


def graph_kernel_ms(name, pd, po, heads, d_head, key):
    """Measured real-block time (ms) of kernel `key` (attn_fwd / attn_bwd / spmm / spmmT) for the rank-0 block of grid
    (P_d, P_o) at head configuration (heads, d_head), or None if that point was not measured."""
    tab = _graph_table(name)
    if not tab:
        return None
    g = tab["grids"].get(f"{pd}x{po}")
    if not g:
        return None
    c = g["configs"].get(f"{heads}x{d_head}")
    return c["real"][key] if c else None


def graph_kernel_ratio(name, heads, d_head, key, cols, d, model_fn):
    """Fallback for unmeasured shapes: measured / model (random-block constants) at the nearest measured K / V footprint
    of the nearest measured head configuration; 1.0 without a table."""
    tab = _graph_table(name)
    if not tab:
        return 1.0
    cfgs = {c for g in tab["grids"].values() for c in g["configs"]}
    if not cfgs:
        return 1.0
    cfg = min(cfgs, key=lambda c: (abs(int(c.split("x")[0]) * int(c.split("x")[1]) - heads * d_head), abs(int(c.split("x")[0]) - heads)))
    h, dh = map(int, cfg.split("x"))
    mb = cols * d * 4 / 2 ** 20
    pts = []
    for g in tab["grids"].values():
        c = g["configs"].get(cfg)
        if c:
            pred = model_fn(g["E"], g["E0"], g["R"], g["C"], h, dh)
            if pred > 0:
                pts.append((abs(math.log(max(c["kv_mb"], 1e-3) / max(mb, 1e-3))), c["real"][key] / pred))
    return min(pts)[1] if pts else 1.0


@dataclass
class Model:
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    tf32: bool = True
    wire_bytes: int = 2         # bf16 gathers / reduce-scatters
    with_gcn: bool = True
    classes: int = 47
    sparse_comm: bool = False   # sparsity-aware collectives (gt3d/dist/sparse_comm.py): see sparse_coll_ms


# ------------------------------------------------------------------------------------------------------------
# cost primitives (ms)
# ------------------------------------------------------------------------------------------------------------
def _interp(table, key):
    """Linear interpolation over the keys of a {d: value-or-tuple} table (clamped at the ends)."""
    ks = sorted(table)
    if key in table:
        return table[key]
    lo = max([k for k in ks if k < key], default=ks[0]); hi = min([k for k in ks if k > key], default=ks[-1])
    if lo == hi:
        return table[lo]
    t = (key - lo) / (hi - lo)
    a, b = table[lo], table[hi]
    if isinstance(a, tuple):
        return tuple(x + t * (y - x) for x, y in zip(a, b))
    return a + t * (b - a)


def _curve_ms(kind, key, chunk_mb):
    """Measured call time of NCCL_CURVES[key][kind] at a per-rank chunk of chunk_mb MB: piecewise linear in the size
    between the measured points (= a local alpha + bytes / beta on every segment), the smallest point as the floor,
    the last segment's slope beyond the largest point.  Times are made non-decreasing in the size first."""
    pts = NCCL_CURVES[key][kind]
    xs = [x for x, _ in pts]
    ys, run = [], 0.0
    for _, y in pts:
        run = max(run, y); ys.append(run)
    if chunk_mb <= xs[0]:
        return ys[0]
    if chunk_mb >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (chunk_mb - xs[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= chunk_mb < xs[i + 1]:
            t = (chunk_mb - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def coll_ms(kind, layout, nbytes, g=None):
    """One collective of the given kind (ag / rs / a2a) in a group of layout (k ranks per node, m nodes); nbytes =
    bytes this rank receives (ring all-gather / reduce-scatter: (g-1)/g of the full tensor).  With the group size g the
    measured size curve of the layout is used (NCCL_CURVES); without it the alpha-beta pair of HW["nccl"]."""
    k, m = layout
    if k * m <= 1 or nbytes <= 0:
        return 0.0
    tab = HW["nccl"]
    exact = (min(k, 4), m)
    key = exact if exact in tab else ((min(k, 4), min(m, 4)) if m > 1 else (min(k, 4), 1))
    # the size curves are indexed by the per-rank chunk and belong to ONE group size: only an exactly measured layout
    # may use them (a 64-node group mapped onto the 4-node curve would be charged 4 chunks instead of 64); unmeasured
    # layouts (beyond 16 nodes) fall back to alpha + bytes / rate of the nearest measured layout, which scales with the
    # bytes this rank receives whatever the group size
    if g is not None and g > 1 and exact in NCCL_CURVES:
        chunk = nbytes / (g - 1) if kind in ("ag", "rs") else nbytes * g / (g - 1)
        return _curve_ms(kind, exact, chunk / 2 ** 20)
    if key not in tab:
        key = (4 if m == 1 else key[0], key[1]) if (4 if m == 1 else key[0], key[1]) in tab else min(tab, key=lambda t: abs(t[0] - key[0]) + abs(t[1] - key[1]))
    alpha, gbs = tab[key][kind]
    return alpha + nbytes / gbs / 1e6


def sparse_share(g: Graph, p: int, exp_deg: int = 3):
    """Share of a gathered block's rows a tile references when the block is split p ways, under the random id order
    (bench/sparse_comm_model.py): the Poisson share of the real attention edges, 1 - exp(-d / p), united with the
    expander floor 1 - (1 - 1/p)^(2 d_e); the same on every rank."""
    if p <= 1:
        return 1.0
    d_real = max(g.e_attn / g.n - 2 * exp_deg, 0.0)
    f_real = 1 - math.exp(-d_real / p)
    f_exp = 1 - (1 - 1 / p) ** (2 * exp_deg)
    return 1 - (1 - f_real) * (1 - f_exp)


def sparse_coll_ms(kind, layout, nbytes, g_size, share, m: Model):
    """coll_ms of a row gather / reduce-scatter, or — with m.sparse_comm and the plan's rule satisfied — its sparse
    form: an all-to-all-v of share x the bytes plus the sender's index_select / receiver's index_add_ (elementwise
    pass over the moved bytes).  The rule mirrors sparse_comm.SparseRowPlan.worthwhile: share <= (a2a / AG of the
    group's node layout) x GT3D_SPARSE_MARGIN."""
    dense = coll_ms(kind, layout, nbytes, g_size)
    if not getattr(m, "sparse_comm", False) or g_size is None or g_size <= 1 or nbytes <= 0:
        return dense
    from gt3d.dist.sparse_comm import A2A_OVER_AG, MARGIN
    k, nodes = layout
    ratio = A2A_OVER_AG.get((min(k, 4), nodes > 1), A2A_OVER_AG[(4, nodes > 1)])
    if share > ratio * MARGIN:
        return dense
    passes = 2 if kind == "ag" else 4                 # gather: index_select; reduce-scatter: permutation + segmented sum
    return coll_ms("a2a", layout, share * nbytes, g_size) + ew_ms(share * nbytes * passes, 2)


def gemm_ms(flops, d, tf32=True, which=0):
    """which: 0 forward, 1 grad_input, 2 grad_W (TFLOPS measured per hidden size)."""
    rate = _interp(HW["gemm"]["tf32" if tf32 else "ieee"], d)[which]
    return flops / rate / 1e9


def ew_ms(nbytes, n_ops=1):
    """Elementwise work: bytes at the measured elementwise rate plus the per-op floor."""
    return nbytes / HW["ew_gbs"] / 1e6 + n_ops * HW["op_us"] / 1e3


L2_BYTES = 32 * 2 ** 20          # A100 L2 is 40 MB; the key / value block (cols x d x 4 B) fits below ~32 MB
L2_FACTOR = dict(fwd=0.6, bwd=0.85)   # per-edge cost in the L2 regime relative to HBM streaming (random block with
                                      # C = 58K rows, bench/hw_constants.py "reddit64": 0.63 forward, 0.85 backward)


def l2_factor(cols, d, direction):
    return L2_FACTOR[direction] if cols * d * 4 <= L2_BYTES else 1.0


TILED_THRESHOLD = 3.5            # edges per row / per column below which the tiled (compact) kernels are dispatched


def _kernel_constants(heads, d_head, direction, regime):
    tab = HW["attn_rows" if regime == "rows" else "attn_compact"]
    key = (heads, d_head)
    if key in tab:
        return tab[key][direction]
    near = min(tab, key=lambda k: abs(k[0] * k[1] - heads * d_head) + 0.01 * abs(k[0] - heads))   # nearest, scaled by H d_h
    scale = (heads * d_head) / (near[0] * near[1])
    return tuple(x * scale for x in tab[near][direction])


# ------------------------------------------------------------------------------------------------------------
# Analytic time of the per-row attention kernels (bench/kernel_theory.py, bench/kernel_l2.py; RESULTS 2026-10-26).
# A program is one row (forward, backward row pass) or one column (backward column pass) of the block for one group of
# at most 4 heads.  It gathers 2w bytes per edge (w = 4 H_g d_h bytes, one head group's row of Q / K / V) in chunks of
# 16 edges and moves a fixed number of bytes for its own row or column; it takes the longer of its transfer time at the
# HBM read bandwidth and its latency floor of H_g tau per chunk (the dependent loads of a chunk; the row pass also adds
# the per-type gate gradient with atomics, hence its larger tau):
#     t_p = max(ceil(d_p / 16) H_g tau,  lambda bytes_p / B)
# Degrees are Poisson(edges / rows) as in a uniform random block; a gather whose rows hit in L2 (share
# min(1, L2 / working set)) costs lambda of its HBM time.  B, tau and lambda are single-GPU micro-benchmarks on
# regular and L2-resident blocks; nothing is fitted to the blocks of a graph or to training runs.
# ------------------------------------------------------------------------------------------------------------
KERNEL_MODEL = "theory"          # "constants": the linear random-block constants HW["attn_rows"] (before 2026-10-26)
THEORY = dict(B_gbs=1440.0, l2_bytes=40 * 2 ** 20, chunk=16, heads_per_program=4,
              tau_ns=dict(fwd=1.29, row=3.43, col=2.53),       # per head and chunk, regular blocks with one edge per row
              lam=dict(fwd=0.37, row=0.73, col=0.62),          # gathered side resident in L2 (8K rows, 64 edges per row)
              # tiled kernel (fewer than TILED_THRESHOLD edges per row / column): 16 rows per program, one-hot gathers;
              # (ns per row, ns per edge) of each pass on regular blocks with 1 and 8 edges per row, {(H, d_h): ...}
              tiled={(4, 32): dict(fwd=(2.08, 0.88), row=(1.20, 2.15), col=(2.04, 1.20)),
                     (2, 32): dict(fwd=(0.43, 0.84), row=(0.77, 0.96), col=(0.93, 0.83)),
                     (1, 32): dict(fwd=(1.00, 0.60), row=(0.84, 0.74), col=(1.11, 0.71)),
                     (8, 32): dict(fwd=(2.12, 2.05), row=(3.14, 5.29), col=(4.55, 2.89))})


def _tiled_ms(n_prog, edges, heads, d_head, which):
    tab = THEORY["tiled"]
    key = (heads, d_head)
    scale = 1.0
    if key not in tab:
        key = min(tab, key=lambda k: abs(k[0] * k[1] - heads * d_head))
        scale = heads * d_head / (key[0] * key[1])
    a, b = tab[key][which]
    return scale * (a * n_prog + b * edges) / 1e6


@lru_cache(maxsize=4096)
def _poisson_pmf(rho):
    hi = int(rho + 12 * math.sqrt(rho) + 30)
    d = np.arange(hi + 1, dtype=float)
    p = np.exp(-rho + d * math.log(max(rho, 1e-300)) - np.array([math.lgamma(x + 1) for x in d]))
    return d, p / p.sum()


def _theory_pass_ms(n_prog, edges, ws_rows, heads, d_head, which):
    """One pass of the per-row kernels over n_prog programs holding `edges` edges, gathering from ws_rows rows."""
    if n_prog <= 0:
        return 0.0
    th = THEORY
    ng = math.ceil(heads / th["heads_per_program"]); hg = min(heads, th["heads_per_program"]); w = 4 * hg * d_head
    hit = min(1.0, th["l2_bytes"] / max(ws_rows * 2 * w, 1.0))
    lam = 1.0 - hit * (1.0 - th["lam"][which])
    if which == "fwd":       # gather k_j, v_j (+ column index, edge type); own: q_i in, acc_i out, l_i, m_i, segment
        e_b, r_b = 2 * w + 8, 2 * w + 8 * hg + 20
    elif which == "row":     # gather k_j, v_j; own: q_i, dO_i in, dq_i out, three per-head scalars
        e_b, r_b = 2 * w + 8, 3 * w + 12 * hg + 20
    else:                    # column pass: gather q_i, dO_i, l_i, m_i, D_i; own: k_j, v_j in, dk_j, dv_j out
        e_b, r_b = 2 * w + 12 * hg + 8, 4 * w + 20
    d, p = _poisson_pmf(round(edges / n_prog, 3 if edges < 10 * n_prog else 1))
    segs = np.maximum(1.0, np.ceil(d / 512))
    lat = np.maximum(1.0, np.ceil(d / th["chunk"])) * hg * th["tau_ns"][which] * 1e-9
    if which == "row":       # chunk-bound at every degree: L2 shortens the chunk (measured on the resident block)
        lat = lat * lam
    t = np.maximum(lat, lam * (d * e_b + segs * r_b) / (th["B_gbs"] * 1e9))
    return ng * n_prog * float((p * t).sum()) * 1e3


def _attn_theory_ms(e, rows, cols, heads, d_head, direction):
    """Analytic kernel time of a uniform random block; a pass with fewer than TILED_THRESHOLD edges per row (column)
    runs the tiled kernel, charged with its per-row (per-column) and per-edge costs measured on regular blocks."""
    tiled_r = rows > 0 and e / rows < TILED_THRESHOLD
    tiled_c = cols > 0 and e / cols < TILED_THRESHOLD
    if direction == "fwd":
        return _tiled_ms(rows, e, heads, d_head, "fwd") if tiled_r else _theory_pass_ms(rows, e, cols, heads, d_head, "fwd")
    t_row = _tiled_ms(rows, e, heads, d_head, "row") if tiled_r else _theory_pass_ms(rows, e, cols, heads, d_head, "row")
    t_col = _tiled_ms(cols, e, heads, d_head, "col") if tiled_c else _theory_pass_ms(cols, e, rows, heads, d_head, "col")
    return t_row + t_col


def _attn_rand_ms(e, rows, cols, heads, d_head, direction, regime="rows"):
    """Kernel time of a uniform random block of this shape (the model without a per-graph table)."""
    if regime == "compact":
        c = _kernel_constants(heads, d_head, direction, "compact")
        return (c[0] * e + c[1] * rows + c[2] * cols) / 1e6
    if KERNEL_MODEL == "theory":
        return _attn_theory_ms(e, rows, cols, heads, d_head, direction)
    f = l2_factor(cols, heads * d_head, direction)
    if direction == "fwd":
        c = _kernel_constants(heads, d_head, "fwd", "rows")
        return (f * c[0] * e + c[1] * rows + c[2] * cols) / 1e6
    cr = _kernel_constants(heads, d_head, "bwd", "rows" if rows == 0 or e / rows >= TILED_THRESHOLD else "compact")
    cc = _kernel_constants(heads, d_head, "bwd", "rows" if cols == 0 or e / cols >= TILED_THRESHOLD else "compact")
    return (f * (cr[0] + cc[0]) / 2 * e + cr[1] * rows + cc[2] * cols) / 1e6


def attn_ms(e, rows, cols, heads, d_head, direction, regime="rows", graph=None, grid=None):
    """One block: forward = row pass; backward = row pass over the CSR (per-edge share + per-row cost) + column pass
    over the CSC (per-edge share + per-column cost).  Each backward pass is dispatched on its own density (edges per row
    / per column vs TILED_THRESHOLD) like triton_kernel.attn_bwd_partial; the per-edge cost is split equally between
    the two passes.  regime="compact": the streamed sub-blocks (rows = non-empty rows).  graph + grid=(P_d, P_o): the
    measured time of that real block if the graph's calibration table has it, else the constants scaled by the
    measured / model ratio of the nearest measured footprint."""
    if graph is not None and regime != "compact":
        if grid is not None:
            t = graph_kernel_ms(graph, grid[0], grid[1], heads, d_head, "attn_" + direction)
            if t is not None:
                return t
        ratio = graph_kernel_ratio(graph, heads, d_head, "attn_" + direction, cols, heads * d_head,
                                   lambda E, E0, R, C, h, dh: _attn_rand_ms(E, R, C, h, dh, direction))
        return ratio * _attn_rand_ms(e, rows, cols, heads, d_head, direction, regime)
    return _attn_rand_ms(e, rows, cols, heads, d_head, direction, regime)


def _spmm_rand_ms(e, rows, cols, d, transpose=False, compact=False):
    c = _interp(HW["spmm_compact" if compact else "spmm"], d)[1 if transpose else 0]
    f = 1.0 if compact else l2_factor(cols if not transpose else rows, d, "bwd" if transpose else "fwd")
    return (f * c[0] * e + c[1] * rows + c[2] * cols) / 1e6


def spmm_ms(e, rows, cols, d, transpose=False, compact=False, graph=None, grid=None):
    """cuSPARSE SpMM A x (or A^T g) on the GCN block; graph + grid as in attn_ms (the table's configuration with the
    same width d is used: the GCN message width is H x d_head of the configuration)."""
    key = "spmmT" if transpose else "spmm"
    if graph is not None and not compact:
        if grid is not None:
            tab = _graph_table(graph)
            g = tab["grids"].get(f"{grid[0]}x{grid[1]}") if tab else None
            if g:
                for cfg, c in g["configs"].items():
                    h, dh = map(int, cfg.split("x"))
                    if h * dh == d:
                        return c["real"][key]
        ratio = graph_kernel_ratio(graph, 1, d, key, cols if not transpose else rows, d,
                                   lambda E, E0, R, C, h, dh: _spmm_rand_ms(E0, R, C, h * dh, transpose))
        return ratio * _spmm_rand_ms(e, rows, cols, d, transpose, compact)
    return _spmm_rand_ms(e, rows, cols, d, transpose, compact)


def placement(pd, po, pi=1, gpn=GPUS_PER_NODE):
    """Layout (k ranks per node, m nodes) of the head, row and column groups.  rank = (r P_o + c) P_i + h (P_i = 1 is
    the row-major 2D grid): the head group is P_i consecutive ranks; the row group (fixed r, h; all c) has stride P_i
    and spans P_o P_i ranks; the column group (fixed c, h; all r) has stride P_o P_i and spans P."""
    def layout(size, stride):
        span = size * stride
        if span <= gpn:                                 # inside one node
            return (size, 1)
        per_node = max(1, gpn // stride)                # members per node when the stride is below the node size
        return (min(per_node, size), max(1, size // min(per_node, size)))
    return dict(head=layout(pi, 1), row=layout(po, pi), col=layout(pd, po * pi))


def cross_node_groups(pd, po, pi=1, gpn=GPUS_PER_NODE):
    """Number of the rank's groups (head, row, column) that span several nodes."""
    return sum(1 for k, m in placement(pd, po, pi, gpn).values() if m > 1)


# The one-cross-node-axis rule (a rank in at most ONE group that spans nodes) was an artefact of unsynchronised
# collective posting (RESULTS 2026-10-17, "two cross-node groups"): with grid2d's pre-collective stream sync (the
# default since 2026-10-17) the shapes it excluded — (2,2,4), (2,1,8), (2,4,2), fused 2x8 at 16 GPUs — land on this
# model within 2-4 %.  ONE_CROSS_NODE_AXIS=True restores the rule (for runs with GT3D_SYNC_COLLECTIVES=0).
ONE_CROSS_NODE_AXIS = False


def admissible(pd, po, pi=1, gpn=GPUS_PER_NODE):
    """True unless ONE_CROSS_NODE_AXIS is set and the rank would take part in two groups that span nodes."""
    return (not ONE_CROSS_NODE_AXIS) or cross_node_groups(pd, po, pi, gpn) <= 1


def nonempty_fraction(lam, kappa=0.8):
    """Share of the (row, sub-block) pairs with at least one edge when a row has lam edges per sub-block on average:
    1 - (1 + lam/kappa)^-kappa for a negative-binomial degree distribution with dispersion kappa (a statistic of the
    GRAPH: kappa = 0.8 reproduces the measured papers100M fractions 11.6 / 20.4 / 33.6 / 50.8 % at lam = 0.14 / 0.28 /
    0.57 / 1.13); kappa -> inf is the Poisson bound 1 - e^-lam."""
    if kappa is None or kappa > 1e6:
        return 1.0 - math.exp(-lam)
    return 1.0 - (1.0 + lam / kappa) ** (-kappa)


# ------------------------------------------------------------------------------------------------------------
# fused 2D layer: ms per rank per layer, forward and backward, with the breakdown
# ------------------------------------------------------------------------------------------------------------
def fused_layer_ms(pd, po, g: Graph, m: Model, gpn=GPUS_PER_NODE):
    P = pd * po
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    R, C, O = g.n / pd, g.n / po, g.n / P
    ea, eg = g.e_attn / P, g.e_gcn / P
    pl = placement(pd, po, 1, gpn)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)       # sparse-communication shares (sparse_coll_ms)
    w = 2 * d + H                                              # width of the reduce-scattered partial [acc | l | gcn]
    f, bw = {}, {}
    # ---- forward
    f["gather"] = sparse_coll_ms("ag", pl["col"], (C - O) * d * b, pd, sh_c, m) + sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m) \
        + ew_ms((C + R) * d * (b + 4) + O * d * b * 2, 4)                       # casts to the wire dtype and back
    f["gemm"] = gemm_ms(2 * d * d * (3 * C + R) if m.with_gcn else 2 * d * d * (2 * C + R), d, m.tf32, 0)
    f["spmm"] = spmm_ms(eg, R, C, d, graph=g.name, grid=(pd, po)) if m.with_gcn else 0.0
    f["attn"] = attn_ms(ea, R, C, H, dh, "fwd", graph=g.name, grid=(pd, po))
    f["combine"] = ew_ms(R * w * 4 * 2 + R * w * (4 + b), 4) + sparse_coll_ms("rs", pl["row"], (R - O) * w * b, po, sh_r, m) \
        + ew_ms(O * w * (b + 4) + O * d * 4 * 2, 3)
    f["tail"] = ew_ms(O * d * 4 * 14, 14) + gemm_ms(8 * O * d * d, d, m.tf32, 0)   # dropouts, LNs, residuals, relu, FFN
    # ---- backward (mirror image)
    bw["tail"] = ew_ms(O * d * 4 * 28, 28) + gemm_ms(8 * O * d * d, d, m.tf32, 1) + gemm_ms(8 * O * d * d, d, m.tf32, 2)
    bw["combine"] = ew_ms(O * w * (4 + b) + O * d * 4 * 3, 4) + sparse_coll_ms("ag", pl["row"], (R - O) * w * b, po, sh_r, m) \
        + ew_ms(R * w * (b + 4) + R * w * 4, 3)
    bw["attn"] = attn_ms(ea, R, C, H, dh, "bwd", graph=g.name, grid=(pd, po))
    bw["spmm"] = spmm_ms(eg, R, C, d, transpose=True, graph=g.name, grid=(pd, po)) if m.with_gcn else 0.0
    fl = 2 * d * d * (3 * C + R) if m.with_gcn else 2 * d * d * (2 * C + R)
    bw["gemm"] = gemm_ms(fl, d, m.tf32, 1) + gemm_ms(fl, d, m.tf32, 2)
    bw["dx"] = ew_ms(C * d * 4 * (6 if m.with_gcn else 4) + C * d * (4 + b), 4) + sparse_coll_ms("rs", pl["col"], (C - O) * d * b, pd, sh_c, m) \
        + ew_ms(R * d * (4 + b), 2) + sparse_coll_ms("rs", pl["row"], (R - O) * d * b, po, sh_r, m) + ew_ms(O * d * (b + 4) * 2 + O * d * 4 * 4, 6)
    return f, bw


def fixed_step_ms(P, g: Graph, m: Model, gpn=GPUS_PER_NODE, layers=None):
    """Per-step work outside the layers: pre-MP (Linear f_in -> d, LN, dropout, L2 normalise), the output head and
    loss, the flat gradient all-reduce (tree over the nodes), the optimiser; forward + backward."""
    O, d = g.n / P, m.hidden
    L = m.layers if layers is None else layers
    pre = gemm_ms(2 * O * g.f_in * d, d, m.tf32, 0) + gemm_ms(2 * O * g.f_in * d, d, m.tf32, 1) + gemm_ms(2 * O * g.f_in * d, d, m.tf32, 2) \
        + ew_ms(O * d * 4 * 20 + O * g.f_in * 4, 20)
    head = gemm_ms(2 * O * d * m.classes * 3, d, m.tf32, 0) + ew_ms(O * m.classes * 4 * 6, 10)
    n_params = L * 12 * d * d + g.f_in * d + d * m.classes
    nodes = max(1, P // gpn)
    lay = (min(P, gpn), nodes)                                   # the flat gradient all-reduce spans every rank
    ar = coll_ms("rs", lay, 2 * n_params * 4) + coll_ms("ag", lay, 1) * math.ceil(math.log2(max(P, 2)))
    opt = 0.3 + ew_ms(n_params * 4 * 12, 10)
    return pre + head + ar + opt


def cpu_floor_ms(m: Model, ops_per_layer=(55, 95)):
    """CPU-side launch / autograd floor of one step (ops per layer counted from the fused layer: ~55 forward ops and
    ~95 backward nodes per layer plus ~60 / ~40 outside the layers)."""
    L = m.layers
    f, b = ops_per_layer
    return ((f * L + 60) * HW["fwd_op_us"] + (b * L + 40) * HW["bwd_node_us"]) / 1e3


def predict_fused_ms(pd, po, g: Graph, m: Model, gpn=GPUS_PER_NODE, breakdown=False):
    """ms per epoch (one full-graph step) of the fused 2D layer model: the GPU timeline of L layers plus the fixed
    per-step work, or the CPU floor if that is longer (launch-bound regime)."""
    f, bw = fused_layer_ms(pd, po, g, m, gpn)
    gpu = m.layers * (sum(f.values()) + sum(bw.values())) + fixed_step_ms(pd * po, g, m, gpn)
    total = max(gpu, cpu_floor_ms(m))
    if breakdown:
        parts = {k: m.layers * v for k, v in f.items()}
        parts.update({"bwd " + k: m.layers * v for k, v in bw.items()})
        parts["fixed"] = fixed_step_ms(pd * po, g, m, gpn)
        return total, parts
    return total


# ------------------------------------------------------------------------------------------------------------
# attention operator only (bench/bench_dist_attention.py): one layer forward + backward, fused family or head grid
# ------------------------------------------------------------------------------------------------------------
def attn_operator_ms(pd, po, pi, g: Graph, m: Model, family="fused", gpn=GPUS_PER_NODE):
    P = pd * po * pi
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    pl = placement(pd, po, pi, gpn)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)
    if family == "fused":
        assert pi == 1
        R, C, O = g.n / pd, g.n / po, g.n / P
        ea = g.e_attn / P
        w = d + H
        fwd = sparse_coll_ms("ag", pl["col"], (C - O) * d * b, pd, sh_c, m) + sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m) + ew_ms((C + R) * d * (b + 4), 4) \
            + gemm_ms(2 * d * d * (2 * C + R), d, m.tf32, 0) + attn_ms(ea, R, C, H, dh, "fwd", graph=g.name, grid=(pd, po)) \
            + ew_ms(R * w * 4 * 2 + R * w * (4 + b), 4) + sparse_coll_ms("rs", pl["row"], (R - O) * w * b, po, sh_r, m) + ew_ms(O * w * (b + 4) + O * d * 8, 3)
        bwd = ew_ms(O * w * (4 + b) + O * d * 8, 4) + sparse_coll_ms("ag", pl["row"], (R - O) * w * b, po, sh_r, m) + ew_ms(R * w * (b + 4), 3) \
            + attn_ms(ea, R, C, H, dh, "bwd", graph=g.name, grid=(pd, po)) + gemm_ms(2 * d * d * (2 * C + R), d, m.tf32, 1) + gemm_ms(2 * d * d * (2 * C + R), d, m.tf32, 2) \
            + ew_ms(C * d * 4 * 4 + C * d * (4 + b), 4) + sparse_coll_ms("rs", pl["col"], (C - O) * d * b, pd, sh_c, m) \
            + ew_ms(R * d * (4 + b), 2) + sparse_coll_ms("rs", pl["row"], (R - O) * d * b, po, sh_r, m) + ew_ms(O * d * (b + 4) * 2, 4)
        return fwd + bwd
    # head grid: owner projections, Ulysses a2a, head-slice gathers, kernel on H / P_i heads
    O = g.n / P
    rows_q, rows_kv = g.n / pd, g.n / po                      # rows of the gathered head slices (row / column block)
    ds = d / pi                                               # width of a head slice
    Hs = max(H // pi, 1)
    ea = g.e_attn / (pd * po)
    w = ds + Hs
    a2a = lambda nb: coll_ms("a2a", pl["head"], nb * (1 - 1 / pi), pi) if pi > 1 else 0.0
    fwd = gemm_ms(3 * 2 * O * d * d, d, m.tf32, 0) + 3 * a2a(O * d * b) + ew_ms(3 * O * d * (b + 4) * 2, 9) \
        + 2 * sparse_coll_ms("ag", pl["col"], (rows_kv - O * pi) * ds * b, pd, sh_c, m) + sparse_coll_ms("ag", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m) \
        + ew_ms((2 * rows_kv + rows_q) * ds * (b + 4), 3) + attn_ms(ea, rows_q, rows_kv, Hs, dh, "fwd", graph=g.name, grid=(pd, po)) \
        + ew_ms(rows_q * w * 4 * 2 + rows_q * w * (4 + b), 4) + sparse_coll_ms("rs", pl["row"], (rows_q - O * pi) * w * b, po, sh_r, m) \
        + ew_ms(O * pi * w * (b + 4) + O * pi * ds * 8, 3) + a2a(O * d * b)
    bwd = a2a(O * d * b) + ew_ms(O * pi * w * (4 + b) + O * pi * ds * 8, 4) + sparse_coll_ms("ag", pl["row"], (rows_q - O * pi) * w * b, po, sh_r, m) \
        + ew_ms(rows_q * w * (b + 4), 3) + attn_ms(ea, rows_q, rows_kv, Hs, dh, "bwd", graph=g.name, grid=(pd, po)) \
        + sparse_coll_ms("rs", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m) + 2 * sparse_coll_ms("rs", pl["col"], (rows_kv - O * pi) * ds * b, pd, sh_c, m) \
        + ew_ms((2 * rows_kv + rows_q) * ds * (4 + b), 3) + 3 * a2a(O * d * b) + ew_ms(3 * O * d * (b + 4), 6) \
        + gemm_ms(3 * 2 * O * d * d, d, m.tf32, 1) + gemm_ms(3 * 2 * O * d * d, d, m.tf32, 2)
    return fwd + bwd


# ------------------------------------------------------------------------------------------------------------
# head-grid model (Grid3D, attention-only Exphormer: gt3d/models/exphormer_dist.DistExphormerLayerHead)
# ------------------------------------------------------------------------------------------------------------
HEAD_OPS = (62, 110)             # forward ops / backward nodes per head layer: the fused layer's counts plus the a2a / gather wrappers


def head_gcn_ms(pd, po, pi, g: Graph, m: Model, gpn=GPUS_PER_NODE):
    """Extra cost per layer of the GCN branch on the head grid (DistExphormerLayerHead with_gcn): the owner's message
    projection, its d / P_i feature slice riding in the K | V all-to-all and column all-gather (extra bytes only), the
    block SpMM on the slice (the block has E_g / (P_d P_o) edges, replicated over the head group), the extra d / P_i
    columns of the reduce-scatter payload and of the all-to-all back; backward mirrored (SpMM^T, reduce-scatter of the
    message-slice gradient over the column group, projection gradients)."""
    P = pd * po * pi
    d, b = m.hidden, m.wire_bytes
    O, Rq, Rkv = g.n / P, g.n / pd, g.n / po
    ds = d / pi
    eg = g.e_gcn / (pd * po)
    pl = placement(pd, po, pi, gpn)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)
    a2a = lambda nb: coll_ms("a2a", pl["head"], nb * (1 - 1 / pi), pi) if pi > 1 else 0.0
    fwd = gemm_ms(2 * O * d * d, d, m.tf32, 0) + a2a(O * d * b) + sparse_coll_ms("ag", pl["col"], (Rkv - O * pi) * ds * b, pd, sh_c, m) \
        + ew_ms(Rkv * ds * (b + 4), 2) + spmm_ms(eg, Rq, Rkv, ds, graph=g.name, grid=(pd, po)) + ew_ms(Rq * ds * 4 * 2, 2) \
        + sparse_coll_ms("rs", pl["row"], (Rq - O * pi) * ds * b, po, sh_r, m) + a2a(O * d * b) + ew_ms(O * d * 4 * 6, 6)
    bwd = ew_ms(O * d * 4 * 6, 6) + a2a(O * d * b) + sparse_coll_ms("ag", pl["row"], (Rq - O * pi) * ds * b, po, sh_r, m) \
        + spmm_ms(eg, Rq, Rkv, ds, transpose=True, graph=g.name, grid=(pd, po)) + ew_ms(Rkv * ds * (4 + b), 2) \
        + sparse_coll_ms("rs", pl["col"], (Rkv - O * pi) * ds * b, pd, sh_c, m) + a2a(O * d * b) \
        + gemm_ms(2 * O * d * d, d, m.tf32, 1) + gemm_ms(2 * O * d * d, d, m.tf32, 2)
    return fwd + bwd


def predict_head_ms(pd, po, pi, g: Graph, m: Model, gpn=GPUS_PER_NODE, breakdown=False):
    """ms per epoch of the model on the head grid (P_d, P_o, P_i): L x (head attention operator incl. the owner
    projections, attn_operator_ms(family="head"), + the GCN branch if m.with_gcn (head_gcn_ms) + the owned-row tail:
    dropout, norm, residual, FFN) + the fixed per-step work, or the CPU floor if that is longer.  The same tail as
    fused_layer_ms, so the two families are comparable."""
    P = pd * po * pi
    O, d = g.n / P, m.hidden
    op = attn_operator_ms(pd, po, pi, g, m, "head", gpn)
    gcn = head_gcn_ms(pd, po, pi, g, m, gpn) if m.with_gcn else 0.0
    tail_f = ew_ms(O * d * 4 * 14, 14) + gemm_ms(8 * O * d * d, d, m.tf32, 0)
    tail_b = ew_ms(O * d * 4 * 28, 28) + gemm_ms(8 * O * d * d, d, m.tf32, 1) + gemm_ms(8 * O * d * d, d, m.tf32, 2)
    fixed = fixed_step_ms(P, g, m, gpn)
    gpu = m.layers * (op + gcn + tail_f + tail_b) + fixed
    total = max(gpu, cpu_floor_ms(m, HEAD_OPS))
    if breakdown:
        return total, {"attention operator": m.layers * op, "gcn": m.layers * gcn, "tail": m.layers * (tail_f + tail_b), "fixed": fixed}
    return total


def peak_memory_head_bytes(pd, po, pi, n, e_attn, hidden, f_in, layers=4, heads=4, kernel_bf16=False, layer_ckpt=False,
                           with_gcn=False, e_gcn=0, breakdown=False):
    """Peak allocated bytes per rank of the attention-only model on the head grid, counted like peak_memory_bytes:
    structures + features + L x saved tensors + the backward transient of one layer.  Saved per layer: the gathered
    head slices Q [N/P_d, d/P_i], K and V [N/P_o, d/P_i] (fp32, or bf16 with kernel_bf16), the kernel's row maxima
    [N/P_d, H/P_i], and HW["saved"]["head_own_f32"] / ["head_own_mask"] owned-chunk units for the owner projections,
    the a2a / combine outputs, dropout, norms and the FFN (counted once with --saved_census, see bench/hw_constants).
    Transient of the last layer's backward: the gathered (dacc, dl) block, dq, dk, dv and the reduce-scatter inputs."""
    P = pd * po * pi
    d, Hs = hidden, max(heads // pi, 1)
    ds = d / pi
    O, Rq, Rkv = n / P, n / pd, n / po
    u_own = O * d * 4
    base = O * f_in * 4 + (e_attn + (e_gcn if with_gcn else 0)) / (pd * po) * 12 + u_own   # features, int32 CSR / CSC, dx
    s = HW["saved"]
    kb = 0.5 if kernel_bf16 else 1.0
    # gathered Q / K / V head slices, the kernel's row maxima m, the combine's normalised output acc_o [P_i O, H/P_i, d_h]
    # (= 1 U_own) and l_o, plus the owned-chunk tensors of the projections / dropout / norms / FFN (census)
    gathered = kb * (Rq + 2 * Rkv) * ds * 4 + Rq * Hs * 4
    owned = u_own + O * heads * 4 + (s["head_own_f32"] + s["head_own_mask"]) * u_own
    transient = 2 * Rq * (ds + Hs) * 4 + Rq * ds * 4 + 2 * Rkv * ds * 4
    if with_gcn:
        # the gathered message slice [N/P_o, d/P_i] (fp32, SpMM input), the owned GCN output / residual / norm input
        # (census constants head_gcn_own_f32 / _mask), and the transient SpMM^T output of the backward
        gathered += Rkv * ds * 4
        owned += (s["head_gcn_own_f32"] + s["head_gcn_own_mask"]) * u_own
        transient += Rkv * ds * 4 + Rq * ds * 4
    per_layer = gathered + owned
    if breakdown:   # bytes by kind: structures / features, gathered blocks, owned rows, backward transient
        if layer_ckpt:
            return dict(base=base, gathered=gathered, owned=layers * u_own + owned, transient=transient)
        return dict(base=base, gathered=layers * gathered, owned=layers * owned, transient=transient)
    if layer_ckpt:
        return base + layers * u_own + per_layer + transient
    return base + layers * per_layer + transient


# ------------------------------------------------------------------------------------------------------------
# ring attention (gt3d/dist/ring.py, attention-only): grid (P, 1); the layer-input chunks rotate around a ring of the
# P ranks with point-to-point hops while every rank computes its sub-block A[r, c] of the chunk it holds
# ------------------------------------------------------------------------------------------------------------
# One ring hop, measured once (bench/ring_p2p.py, allocation 60313340, 2026-10-26, NCCL 2.27.3 with
# NCCL_NCHANNELS_PER_NET_PEER=4 / NCCL_P2P_NET_CHUNKSIZE=1M as in scripts/multinode_train.sh): ms per hop, max over
# ranks, of a batched {isend to rank+1, irecv from rank-1} per buffer, per buffer size in MB, with one buffer in flight
# (forward: the x chunk) and two (backward: the x chunk and the dx accumulator).  "intra": a ring inside one node
# (NVLink, ~79 GB/s per rank); "inter": a ring over several nodes, which every step waits for at its cross-node hops
# (one NIC per hop, ~23.7 GB/s per rank whether one or two buffers share it).
P2P_CURVES = {
    "intra": {1: [(1, 0.103), (4, 0.105), (16, 0.273), (32, 0.484), (64, 0.901), (128, 1.690)],
              2: [(1, 0.148), (4, 0.208), (16, 0.541), (32, 0.950), (64, 1.772), (128, 3.387)]},
    "inter": {1: [(1, 0.168), (4, 0.193), (16, 0.720), (32, 1.438), (64, 2.832), (128, 5.636)],
              2: [(1, 0.165), (4, 0.565), (16, 1.418), (32, 2.822), (64, 5.792), (128, 11.210)]},
}
RING_OPS = (8, 18)               # forward ops / backward ops per ring step (hop, cast, K / V projections, kernel, adds)


def _points_ms(pts, mb):
    """Piecewise linear in the size between measured (MB, ms) points, the smallest point as the floor and the last
    segment's slope beyond the largest one (as _curve_ms)."""
    xs = [x for x, _ in pts]
    ys, run = [], 0.0
    for _, y in pts:
        run = max(run, y); ys.append(run)
    if mb <= xs[0]:
        return ys[0]
    if mb >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (mb - xs[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= mb < xs[i + 1]:
            return ys[i] + (mb - xs[i]) / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i])
    return ys[-1]


def p2p_ms(nbytes, P, bufs=1, gpn=GPUS_PER_NODE):
    """One ring hop of `bufs` buffers of nbytes each in a ring of P ranks (0 for P = 1)."""
    if P <= 1 or nbytes <= 0:
        return 0.0
    return _points_ms(P2P_CURVES["intra" if P <= gpn else "inter"][bufs], nbytes / 2 ** 20)


def ring_layer_ms(P, g: Graph, m: Model, gpn=GPUS_PER_NODE):
    """One attention-only ring layer per rank, forward and backward (dicts of ms).  P steps over the sub-blocks
    A[r, c] of N/P rows, N/P columns and E/P^2 edges (on average); step s projects K and V of the chunk it holds (the
    ring sends x, so every hop re-projects) and runs the kernel on its sub-block while the chunk travels on: hop s
    overlaps step s, so a step lasts the longer of the two.  Backward: the chunks travel again together with the dx
    accumulators (two buffers per hop); per step K, V are re-projected, the kernel's backward runs on the sub-block and
    the projection gradients go to dW_k, dW_v and to the accumulator; the owner's last accumulator hop is exposed.
    Q is projected (and its gradients formed) once on the owned rows; the owned-row tail is the 2D layer's."""
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    O = g.n / P
    e_sub = g.e_attn / P / P
    w = d + H                                                   # packed partial [acc | l]
    hop1, hop2 = p2p_ms(O * d * b, P, 1, gpn), p2p_ms(O * d * b, P, 2, gpn)
    kv = gemm_ms(2 * 2 * O * d * d, d, m.tf32, 0)
    f, bw = {}, {}
    f["q"] = gemm_ms(2 * O * d * d, d, m.tf32, 0)
    step_f = kv + attn_ms(e_sub, O, O, H, dh, "fwd", graph=g.name, grid=(P, P)) + ew_ms(O * d * (b + 4) + 3 * O * w * 4, 4)
    f["steps"] = (P - 1) * max(step_f, hop1) + step_f
    f["tail"] = ew_ms(O * d * 4 * 14, 14) + gemm_ms(8 * O * d * d, d, m.tf32, 0)
    step_b = kv + attn_ms(e_sub, O, O, H, dh, "bwd", graph=g.name, grid=(P, P)) \
        + gemm_ms(2 * 2 * O * d * d, d, m.tf32, 1) + gemm_ms(2 * 2 * O * d * d, d, m.tf32, 2) \
        + ew_ms(3 * O * d * (b + 4) + 4 * O * d * 4, 8)
    bw["steps"] = (P - 1) * max(step_b, hop2) + step_b + hop1
    bw["q"] = gemm_ms(2 * O * d * d, d, m.tf32, 0) + gemm_ms(2 * O * d * d, d, m.tf32, 1) + gemm_ms(2 * O * d * d, d, m.tf32, 2)
    bw["tail"] = ew_ms(O * d * 4 * 28, 28) + gemm_ms(8 * O * d * d, d, m.tf32, 1) + gemm_ms(8 * O * d * d, d, m.tf32, 2)
    return f, bw


def predict_ring_ms(P, g: Graph, m: Model, gpn=GPUS_PER_NODE, breakdown=False):
    """ms per epoch of the attention-only model on the ring layer: L x ring_layer_ms + the fixed per-step work, or the
    CPU floor (RING_OPS per step on top of the 2D layer's counts) if that is longer."""
    f, bw = ring_layer_ms(P, g, m, gpn)
    fixed = fixed_step_ms(P, g, m, gpn)
    gpu = m.layers * (sum(f.values()) + sum(bw.values())) + fixed
    total = max(gpu, cpu_floor_ms(m, (55 + RING_OPS[0] * P, 95 + RING_OPS[1] * P)))
    if breakdown:
        parts = {k: m.layers * v for k, v in f.items()}
        parts.update({"bwd " + k: m.layers * v for k, v in bw.items()})
        parts["fixed"] = fixed
        return total, parts
    return total


def peak_memory_ring_bytes(P, n, e_attn, hidden, f_in, layers=4, heads=4, wire_bytes=2, breakdown=False):
    """Peak allocated bytes per rank of the attention-only ring model, counted like peak_memory_head_bytes: structures
    + features + L x saved tensors + the working set of one backward step.  No tensor of more than N/P rows exists.
    Saved per layer (the ring Function keeps only its input; the packed partial [acc | l] feeds the tail):
    HW["saved"]["ring_own_f32"] / ["ring_own_mask"] owned-chunk units (--saved_census).  Backward step: q, dq, dacc,
    dl, K, V, dK, dV and dx of the step's chunk (fp32), and the x / dx buffers in flight (wire dtype, two of each)."""
    d, H, b = hidden, heads, wire_bytes
    O = n / P
    u_own = O * d * 4
    base = O * f_in * 4 + e_attn / P * 12 + u_own                    # features, int32 CSR / CSC of the P sub-blocks, dx
    s = HW["saved"]
    per_layer = (s["ring_own_f32"] + s["ring_own_mask"]) * u_own
    transient = 2 * u_own + O * (d + H) * 4 + 5 * u_own + 4 * O * d * b
    if breakdown:   # no gathered block: the ring holds at most one N/P-row chunk of other ranks at a time
        return dict(base=base, gathered=0.0, owned=layers * per_layer, transient=transient)
    return base + layers * per_layer + transient


def candidate_grids_3d(world, heads, gpn=GPUS_PER_NODE, max_pi=None):
    """(P_d, P_o, P_i) with P_i | heads and P_i | world; max_pi caps the head axis (None: up to heads)."""
    out = []
    for pi in range(1, min(heads, world, max_pi or heads) + 1):
        if heads % pi or world % pi:
            continue
        q = world // pi
        out += [(pd, q // pd, pi) for pd in range(1, q + 1) if q % pd == 0]
    return out


def select_grid_head(world, n, e_attn, hidden, f_in, layers=4, heads=None, tf32=True, gpn=GPUS_PER_NODE,
                     capacity_gb=40.0, headroom=0.92, kernel_bf16=False, max_pi=None, with_gcn=False, e_gcn=0, graph=None):
    """Fastest predicted head grid (P_d, P_o, P_i) of the model (with the GCN branch if with_gcn) among the
    memory-feasible ones; returns (best or None, ranked [(pd, po, pi, ms, GB)]).  (P_d, P_o, 1) = the 2D grid with
    projected gathers."""
    heads = heads if heads else {80: 2, 128: 4, 256: 8}.get(hidden, 4)
    g = Graph(n, e_attn, e_gcn, f_in, name=graph)
    m = Model(hidden, heads, layers, tf32, with_gcn=with_gcn)
    ranked = []
    for pd, po, pi in candidate_grids_3d(world, heads, gpn, max_pi):
        mem = peak_memory_head_bytes(pd, po, pi, n, e_attn, hidden, f_in, layers, heads, kernel_bf16, with_gcn=with_gcn, e_gcn=e_gcn) / GB
        ranked.append((pd, po, pi, predict_head_ms(pd, po, pi, g, m, gpn), mem))
    ranked.sort(key=lambda r: r[3])
    fits = [r for r in ranked if r[4] <= capacity_gb * headroom]
    return (fits[0] if fits else None), ranked


# ------------------------------------------------------------------------------------------------------------
# streamed layer (column block in n_sub sub-blocks, recomputation in backward, compact kernels, overlap)
# ------------------------------------------------------------------------------------------------------------
def streamed_layer_ms(pd, po, n_sub, g: Graph, m: Model, gpn=GPUS_PER_NODE, overlap=True):
    P = pd * po
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    R, C, O = g.n / pd, g.n / po, g.n / P
    cs = C / n_sub                                            # rows of one column sub-block
    ea, eg = g.e_attn / P / n_sub, g.e_gcn / P / n_sub        # edges per sub-block
    lam_a, lam_g = ea / R, eg / R
    pairs_a = nonempty_fraction(lam_a, g.kappa) * R           # non-empty rows per sub-block (attention)
    pairs_g = nonempty_fraction(lam_g, g.kappa) * R
    pl = placement(pd, po, 1, gpn)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)       # sparse-communication shares (sparse_coll_ms)
    w = 2 * d + H
    out = {}
    rowside = sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m) + ew_ms(R * d * (b + 4), 2) + gemm_ms(2 * R * d * d, d, m.tf32, 0)
    t_ag = sparse_coll_ms("ag", pl["col"], (cs - O / n_sub) * d * b, pd, sh_c, m)
    proj = ew_ms(cs * d * (b + 4), 2) + gemm_ms(3 * 2 * cs * d * d, d, m.tf32, 0)
    spmm = spmm_ms(eg, pairs_g, cs, d, compact=True) + ew_ms(pairs_g * d * 4 * 3, 2)      # DCSR SpMM + index_add
    attn = attn_ms(ea, pairs_a, cs, H, dh, "fwd", regime="compact")
    comp = proj + spmm + attn
    exposed = n_sub * (max(0.0, t_ag - comp) if overlap else t_ag) + (t_ag if overlap else 0.0)
    rs = ew_ms(R * w * (4 + b), 2) + sparse_coll_ms("rs", pl["row"], (R - O) * w * b, po, sh_r, m) + ew_ms(O * w * (b + 4), 2)
    out["rowside"], out["exposed"], out["proj"], out["spmm"], out["attn"], out["rs"] = rowside, exposed, n_sub * proj, n_sub * spmm, n_sub * attn, rs
    out["tail"] = ew_ms(O * d * 4 * 14, 14) + gemm_ms(8 * O * d * d, d, m.tf32, 0)
    f_stream = rowside + exposed + n_sub * comp + rs
    # backward: tail (recompute + grads), gathers of (dacc, dl, dagg), recompute of the stream, kernel / GEMM backward,
    # dx reduce-scatters per sub-block (overlapped with the next sub-block), dq -> dx_row
    bwd_tail = out["tail"] + ew_ms(O * d * 4 * 28, 28) + gemm_ms(8 * O * d * d, d, m.tf32, 1) + gemm_ms(8 * O * d * d, d, m.tf32, 2)
    dgath = 3 * coll_ms("ag", pl["row"], (R - O) * (w / 3) * b, po) + ew_ms(R * w * (b + 4), 3)
    attn_b = attn_ms(ea, pairs_a, cs, H, dh, "bwd", regime="compact")
    spmm_b = spmm_ms(eg, cs, R, d, transpose=True, compact=True)
    gemm_b = gemm_ms(3 * 2 * cs * d * d, d, m.tf32, 1) + gemm_ms(3 * 2 * cs * d * d, d, m.tf32, 2)
    dx_ew = ew_ms(cs * d * 4 * 6 + cs * d * (4 + b), 6)
    t_rs = sparse_coll_ms("rs", pl["col"], (cs - O / n_sub) * d * b, pd, sh_c, m)
    comp_b = proj + attn_b + spmm_b + gemm_b + dx_ew
    exposed_b = n_sub * (max(0.0, t_ag + t_rs - comp_b) if overlap else t_ag + t_rs) + (t_ag + t_rs if overlap else 0.0)
    dq = gemm_ms(2 * R * d * d, d, m.tf32, 1) + gemm_ms(2 * R * d * d, d, m.tf32, 2) + ew_ms(R * d * (4 + b) * 2, 3) \
        + sparse_coll_ms("rs", pl["row"], (R - O) * d * b, po, sh_r, m)
    out["bwd"] = bwd_tail + dgath + rowside + exposed_b + n_sub * comp_b + dq + ew_ms(O * d * 4 * 4, 4)
    out["fwd"] = f_stream + out["tail"]
    return out


def predict_streamed_ms(pd, po, n, e_attn, e_gcn, hidden, n_sub, heads=4, layers=4, f_in=128, tf32=True, overlap=True,
                        gpn=GPUS_PER_NODE, kappa=0.8):
    g = Graph(n, e_attn, e_gcn, f_in, kappa)
    m = Model(hidden, heads, layers, tf32)
    o = streamed_layer_ms(pd, po, n_sub, g, m, gpn, overlap)
    return max(layers * (o["fwd"] + o["bwd"]) + fixed_step_ms(pd * po, g, m, gpn), cpu_floor_ms(m) * (1 + 0.5 * n_sub / 8))


# ------------------------------------------------------------------------------------------------------------
# memory (bytes per rank), counted
# ------------------------------------------------------------------------------------------------------------
def peak_memory_bytes(pd, po, n, e_attn, e_gcn, hidden, f_in, layers=4, n_sub=1, streamed=False, rs_bf16=True,
                      layer_ckpt=False, heads=4, kernel_bf16=False, breakdown=False):
    """Peak allocated bytes per rank.
    fused, no checkpoint: structures + features + L x saved tensors (census multipliers) + the backward transient of
      the last layer (the gathered (dacc, dl, dagg) block, dq, dk, dv, the three dx_col pieces and their sum).
    fused, layer checkpoint: only the layer inputs are kept; the recomputed layer's saved tensors + transient appear once.
    streamed: the owned-row tail per layer + one layer's working set (row block in the wire dtype, Q, the packed
      partial, one sub-block in flight (+1 prefetched), K/V/GCN projections of a sub-block) + per-sub-block
      structures (~17 B per (row, sub-block) pair for the DCSR / tile plans, 12 B per edge)."""
    P = pd * po
    d, H = hidden, heads
    R, C, O = n / pd, n / po, n / P
    u_own, u_row, u_col = O * d * 4, R * d * 4, C * d * 4
    base = O * f_in * 4 + (e_attn + e_gcn) / P * 12 + u_own                  # features, int32 CSR/CSC/GCN structures, dx of the owned rows
    s = HW["saved"]
    if streamed:
        cs = C / max(n_sub, 1)
        tail = layers * (u_own + O * (2 * d + H) * 4)                                 # layer input + packed RS output (tail checkpoint)
        kb = 0.5 if kernel_bf16 else 1.0
        sub = 2 * cs * d * (2 + 4) + (2 * kb + 1) * cs * d * 4                        # 2 sub-blocks in flight (wire + fp32), K / V / GCN projections
        work_fwd = R * d * 2 + kb * u_row + R * (2 * d + H) * 4 + sub                 # x_row (wire), q, packed partial
        work_bwd = R * d * 2 + u_row + R * (2 * d + H) * 4 + u_row + sub + cs * d * 4 * 3   # + dacc / dl / dagg, dq, dk / dv / dx of a sub-block
        structs = n_sub * R * 17 + (e_attn + e_gcn) / P * 4                           # per-sub-block crow / DCSR / tile plans
        return base + tail + max(work_fwd, work_bwd) + structs
    # kernel_bf16: K, V (2 of the 3 U_col) and Q (1 of the 2 U_row) are saved in bf16
    kb = 0.5 if kernel_bf16 else 1.0
    gathered = (1.0 + 2.0 * kb) * u_col + (s["row"] - 1.0 + kb) * u_row
    owned = (s["own_f32"] + s["own_mask"]) * u_own
    per_layer = gathered + owned
    # backward of the last layer while every layer's saved tensors are alive: the gathered (dacc, dl, dagg) block, dq
    # and two column-block-sized gradients live at once (measured 2 U_col + 2 U_row on 2x2 / 64x1 products)
    transient = R * (2 * d + H) * 4 + u_row + 2 * u_col
    if breakdown:   # bytes by kind: structures / features, gathered blocks, owned rows, backward transient
        if layer_ckpt:
            return dict(base=base, gathered=gathered, owned=layers * u_own + owned, transient=transient)
        return dict(base=base, gathered=layers * gathered, owned=layers * owned, transient=transient)
    if layer_ckpt:
        return base + layers * u_own + per_layer + transient
    return base + layers * per_layer + transient


def feasible(pd, po, n, e_attn, e_gcn, hidden, f_in, layers=4, n_sub=1, streamed=False, rs_bf16=True,
             capacity_gb=40.0, headroom=0.92, heads=4):
    return peak_memory_bytes(pd, po, n, e_attn, e_gcn, hidden, f_in, layers, n_sub, streamed, rs_bf16, heads=heads) <= capacity_gb * headroom * GB


# ------------------------------------------------------------------------------------------------------------
# grid selection
# ------------------------------------------------------------------------------------------------------------
def attention_edges(num_nodes, num_edges, exp_deg):
    """Attention edges of the Exphormer graph: original (directed count) + self loops + undirected expander edges."""
    return num_edges + num_nodes + 2 * exp_deg * num_nodes


def candidate_grids(world):
    return [(pd, world // pd) for pd in range(1, world + 1) if world % pd == 0]


def select_grid(world, n, e, hidden, gpn=GPUS_PER_NODE, max_po="node", e_gcn=None, heads=None, f_in=128, layers=4, tf32=True):
    """Fastest predicted (P_d, P_o) of the fused layer for `world` GPUs; returns (pd, po, predicted_ms, ranked list).
    max_po="node" keeps the row group inside a node (P_o <= gpn: the node-filling rule); None considers every divisor."""
    g = Graph(n, e, e_gcn if e_gcn is not None else int(e * 0.9), f_in)
    m = Model(hidden, heads if heads else {80: 2, 128: 4, 256: 8}.get(hidden, 4), layers, tf32)
    cands = candidate_grids(world)
    if max_po == "node":
        cands = [c for c in cands if admissible(c[0], c[1], 1, gpn)] or cands
    elif max_po is not None:
        cands = [c for c in cands if c[1] <= max_po] or cands
    ranked = sorted(((pd, po, predict_fused_ms(pd, po, g, m, gpn)) for pd, po in cands), key=lambda r: r[2])
    pd, po, ms = ranked[0]
    return pd, po, ms, ranked


def select_grid_mem(world, n, e_attn, e_gcn, hidden, f_in, layers=4, n_sub=1, streamed=False, rs_bf16=True,
                    gpn=GPUS_PER_NODE, heads=None, tf32=True, capacity_gb=40.0, headroom=0.92, graph=None):
    """Fastest predicted grid among the memory-feasible ones (None if no grid fits); ranked list [(pd, po, ms, GB)]."""
    heads = heads if heads else {80: 2, 128: 4, 256: 8}.get(hidden, 4)
    ranked = []
    for pd, po in candidate_grids(world):
        if not admissible(pd, po, 1, gpn):
            continue
        mem = peak_memory_bytes(pd, po, n, e_attn, e_gcn, hidden, f_in, layers, n_sub, streamed, rs_bf16, heads=heads) / GB
        if streamed:
            ms = predict_streamed_ms(pd, po, n, e_attn, e_gcn, hidden, n_sub, heads, layers, f_in, tf32, True, gpn)
        else:
            ms = predict_fused_ms(pd, po, Graph(n, e_attn, e_gcn, f_in, name=graph), Model(hidden, heads, layers, tf32), gpn)
        ranked.append((pd, po, ms, mem))
    ranked.sort(key=lambda r: r[2])
    fits = [r for r in ranked if r[3] <= capacity_gb * headroom]
    return (fits[0] if fits else None), ranked


def select_grid_streamed(world, n, e_attn, e_gcn, hidden, f_in, gpn=GPUS_PER_NODE, capacity_gb=40.0, headroom=0.9,
                         n_sub_choices=(4, 8, 16, 32, 64, 128), heads=None, layers=4, tf32=True):
    """Fastest predicted (P_d, P_o, n_sub) of the streamed layer among the memory-feasible ones (P_o <= GPUs per node,
    the smallest n_sub that fits for every grid).  Returns (best or None, ranked [(pd, po, n_sub, ms, GB)])."""
    heads = heads if heads else {64: 4, 80: 2, 128: 4, 256: 8}.get(hidden, 4)
    ranked = []
    for pd, po in candidate_grids(world):
        if not admissible(pd, po, 1, gpn):
            continue
        for n_sub in n_sub_choices:
            mem = peak_memory_bytes(pd, po, n, e_attn, e_gcn, hidden, f_in, layers, n_sub, streamed=True, heads=heads) / GB
            if mem <= capacity_gb * headroom:
                ranked.append((pd, po, n_sub, predict_streamed_ms(pd, po, n, e_attn, e_gcn, hidden, n_sub, heads, layers, f_in, tf32, True, gpn), mem))
                break
    ranked.sort(key=lambda r: r[3])
    return (ranked[0] if ranked else None), ranked
