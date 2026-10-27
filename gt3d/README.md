# gt3d — scalable full-graph training of structure-masked sparse graph attention

Work-in-progress research code.

## Layout

| path | content | status |
|---|---|---|
| `gt3d/attention/ref.py` | edge-list reference: dot / gated / biased scores, softmax / clamp-exp norms | done, fp64 spec |
| `gt3d/attention/backend_torch.py` | Backend A: CSR + segment_csr forward, explicit backward (no `index_put_`), row-chunked | done, tested vs ref |
| `gt3d/attention/triton_kernel.py` | Backend B prototype: fused Triton kernel, one program per row, all heads; atomic dK/dV | done, tested vs A |
| `gt3d/models/exphormer_ref.py` | official Exphormer `GCN+Exphormer` MultiModel without GraphGym + state-dict mapping | done, official test pending |
| `gt3d/data/expander.py` | Random-d expander edges (no lambda_2 loop) | done |
| `gt3d/dist/ring.py` | ring attention (1D context parallelism: (P, 1) row partition, input chunks rotate by NCCL P2P, K/V projected per hop, recomputed in backward; memory O(N/P)) — motivation baseline, `train_dist.py --grid P 1 --ring` | done, fp64-tested vs reference |
| `gt3d/dist/ring_kv.py` | ring attention with K/V rotation (Ring Attention as published): owner projects K/V once, the packed [K \| V] chunk rotates by NCCL P2P (P-1 hops of 2 x chunk x hidden), dQ local, the dK/dV accumulator rotates in backward; attention only (no GCN branch), `bench/bench_dist_attention.py --strategy ring_kv`, `tests/test_dist_attention.py --ring_kv` | done, fp64/fp32-tested on 4/2/1 ranks (`scripts/run_ringkv_tests.sh`) |
| `gt3d/dist/grid3d.py` + `sparse_attention.py::DistHeadSparseAttention` | head axis: Grid3D (P_d, P_o, P_i), rank = (r*P_o + c)*P_i + h; owners project Q/K/V, Ulysses all-to-all rows <-> heads inside the head group, all-gather of the head slices over the column / row groups, block kernel on H/P_i heads, reduce-scatter combine, all-to-all back. (P_d, P_o, 1) = 2D with projected gathers, (1, 1, P) = pure head parallelism (GP-A2A / TorchGT, P <= H). `bench/bench_dist_attention.py --strategy head --grid P_d P_o P_i` | done, fp64/fp32-tested on 6 grids (`scripts/run_head_tests.sh`) |
| `gt3d/dist/sparse_comm.py` | sparsity-aware collectives: `SparseRowPlan` = the rows of a gathered block a rank needs (non-empty columns of its tile for the K/V side, non-empty rows for the Q side / partials), exchanged once inside the group; gather = index_select + all-to-all-v, reduce-scatter = all-to-all-v + index_add_ (autograd pairs, async variants); tiles / GCN blocks remapped once to the compact index spaces (`AttentionBlock.compacted`, `layers.GCNBlockView`, `exphormer_dist.build_sparse_comm`) | done, fp64-tested on fused / streamed / head layers (`scripts/run_sparse_comm_tests.sh`) |
| `tests/` | `test_backend_torch.py`, `test_triton_kernel.py`, `test_ref_vs_official.py` | first two pass |
| `bench/bench_attention_single_gpu.py` | one attention layer on ogbn-arxiv: official scatter vs A vs Triton | running |
| `scripts/run_on_node.sh` | env + `srun --jobid ... bash scripts/run_on_node.sh <cmd>` wrapper | |

## Environment

`source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate` (torch 2.8.0+cu128, PyG 2.6.1,
torch_scatter, triton 3.4).  `export DGL_DISABLE_GRAPHBOLT=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
Anything that imports `graphgps` (official Exphormer at `../../baseline/Exphormer`) is slow to
import (minutes on Lustre) and currently deadlocks at import on Perlmutter — see
`logs/`.

## Running

```bash
# tests that do not need graphgps (seconds)
python tests/test_backend_torch.py
python tests/test_triton_kernel.py          # needs a GPU
# single-GPU attention benchmark on ogbn-arxiv
python bench/bench_attention_single_gpu.py --heads 2 --head_dim 40 [--global_token] [--dtype bf16]
```

## Choosing the grid

`gt3d/dist/grid_select.py` holds the white-box performance / memory model: every term is a count (bytes moved per
collective, GEMM FLOPs, edges / rows / columns of the kernels, elementwise passes, saved tensors) times a constant
measured once in isolation (`bench/hw_constants.py`: HBM, elementwise, launch, GEMM TFLOPS, NCCL alpha-beta per group
layout "k ranks per node x m nodes"; `bench/kernel_constants.py`: per-edge / per-row / per-column cost of the attention
and SpMM kernels; `train_dist.py --saved_census`: the tensors autograd keeps).  Nothing is fitted to training runs;
`bench/validate_model.py` compares the predictions with the measured tables of `bench/model_validation_data.py`
(ranking of the grids per GPU count, component breakdown, memory).  `train_dist.py --grid_auto` picks (P_d, P_o) for the
world size from N, the edge counts, hidden size and heads and prints the prediction.  The model assumes the row-major
rank layout, the fused layer, bf16 gathers and TF32 matmuls; `attn_operator_ms` also covers the head grids (P_d, P_o, P_i).

`--stream_sub n` selects the streamed layer (column block gathered in n sub-blocks and recomputed in backward; memory
O(N/P_d + N/(P_o n)) instead of O(N/P_o + N/P_d)), which is what makes papers100M fit at 64-128 GPUs.  `--grid_auto`
then also applies the memory model (`peak_memory_bytes`) and picks the fastest grid predicted to fit in 40 GB.

`bench/torchgt_batch_bench.py --grid 2 2 --fused [--comm_bf16 --rs_bf16]` runs TorchGT's Graphormer with the fused
communication blocks (`tests/test_graphormer_fused.py` checks them against the baseline blocks and the single-GPU model).

`gt3d/attention/triton_tiled.py` holds row-tiled versions of the three kernels for blocks with few edges per row
(streamed sub-blocks); `attn_fwd_partial`/`attn_bwd_partial` pick them automatically below `GT3D_TILED_THRESHOLD`
(default 3.5 edges per row; the backward's row pass and column pass are dispatched separately, on edges per row and
edges per column).  `tests/test_triton_tiled.py` compares them with the per-row kernels.  The per-row kernels process
`GT3D_HEADS_PER_PROGRAM` heads per program (default 4: an 8-head model runs as two head-group passes over slices of the
[n, H, D] tensors; -22..-32 % on the backward).  `train_dist.py --kernel_bf16` stores Q / K / V in bf16 for the kernels
(fp32 accumulation): -8..-12 % peak memory at +2..+4 % epoch time, accuracy unchanged on arxiv (opt-in).

`--grid P 1 --ring` runs the ring-attention layer (`gt3d/dist/ring.py`): the 1D row partition of GP-AG, but the input chunks travel
around the ring of ranks (P-1 point-to-point hops, overlapped with the sub-block kernels) instead of being all-gathered, and the
backward re-rotates them (Ring Attention style), so no O(N) tensor exists per rank.  It is a motivation baseline for the paper's
comparison of context-parallel strategies (head parallel / all-gather / ring / 2D); `scripts/sbatch_ring_motivation.sh` runs
ring vs GP-AG vs the best 2D grid on 4-16 GPUs.  `--checkpoint` checkpoints only the owned-row tail, as for the streamed layer.

`bench/bench_dist_attention.py --strategy ring_kv` (attention only) is the same ring with K/V instead of x travelling
(`gt3d/dist/ring_kv.py`, Ring Attention as published): twice the bytes per hop (hidden = H d) but no per-hop projection
GEMMs and no per-hop dW_k / dW_v / dx GEMMs in the backward.  `scripts/run_attn_ringkv.sh <jobid>` compares ring_kv, ring,
GP-AG and the 2D grid on 4-16 GPUs with compute-only repeats of both rings.
`--strategy ring_kv2` is its bidirectional form (`DistRingAttentionKV(bidir=True)`): half of every chunk's rows travel
clockwise and half counter-clockwise, so each hop uses both directions of every cross-node link with the same bytes per
rank and one kernel per step (`split_block_bidir` builds the per-step sub-blocks); `scripts/run_ringkv2_tests.sh` /
`scripts/run_attn_ringkv2.sh <jobid>`.  The hops only reach the link bandwidth with `NCCL_NCHANNELS_PER_NET_PEER=4` and
`NCCL_P2P_NET_CHUNKSIZE=1048576` (exported by `scripts/multinode_train.sh`; `scripts/run_ringkv2_nccl.sh` is the sweep).

The head-parallel study (`scripts/run_attn_head.sh <jobid>`, attention only) runs the fused 2D reference and every Grid3D shape
(P_d, P_o, P_i) with P_i | H on 4 / 8 / 16 GPUs; `scripts/sbatch_attn_head_scale.sh` (sbatch -N 8 or -N 16, optional
SUBSET=products|reddit for the debug QOS) continues it at 32 / 64 GPUs with the node-filling shapes (P_o x P_i = 4) plus
compute-only repeats; `bench/parse_attn_head.py logs/attn_head_*.log` tabulates the ATTN lines.

`--auto_ckpt` switches activation checkpointing on when the memory model predicts more than 34 GB per GPU (products
with hidden >= 128 on 4 GPUs needs it).

`--sparse_comm` (fused, streamed and head-grid layers; `gt3d/dist/sparse_comm.py`) replaces the all-gathers by
all-to-all-v's that move only the rows a rank's tile references (K/V side: the non-empty columns of A[r, c]; Q side and
the partials' reduce-scatter: its non-empty rows) and the reduce-scatters by the transpose, with static index sets
exchanged once at construction and the CSRs remapped once to the compact index spaces.  Under the random id order the
share of rows needed is 1 - exp(-d_attn / P_d) and the same on every rank (products 59 / 20 / 5 %, papers100M 43 / 13 /
3 % at P_d = 64 / 256 / 1024).  Because NCCL's all-to-all runs at ~0.9x the all-gather rate per rank when the group has
one rank per node but at 0.4x / 0.17x when it spans whole nodes (the ring all-gather's intra-node hops), a collective
goes sparse only when its share is below that ratio x `GT3D_SPARSE_MARGIN` (0.9); `--sparse_threshold 1.0` forces every
collective sparse (tests), `0` none.  The trainer prints the shares and decisions (`[sparse_comm]`) and the STEADY line
carries `sparse=r<0|1>c<0|1>`.  `bench/bench_sparse_comm.py` times dense vs sparse on the real tiles and the primitive's
cost curve vs share; `bench/bench_dist_attention.py --sparse_comm` is the attention-operator pairing with `--no_comm`;
`scripts/sbatch_sparse_comm.sh` (4 / 16 nodes) is the multi-node validation.

`--no_gcn` drops the parallel GCN branch (attention-only Exphormer, GraphGPS `local_gnn_type: None`; plain and fused
layers): within 0.1-0.75 points of the hybrid on arxiv / Reddit / products and 17-23% faster per epoch
(`scripts/run_attn_only_accuracy.sh`, RESULTS.md 2026-10-13).  `--no_attn` is the GCN-only arm (plain layer only;
`scripts/run_gcn_only_accuracy.sh`); the consolidated GCN+GT / GT / GCN / TorchGT table is at the end of that section.  TorchGT (paper-faithful copy, `--official_split`) on the OGB official splits: arxiv 57.3, products 53.1 (RESULTS.md, 2026-10-13 continued).

## Dependencies on Plexus and AxoNN (used unmodified)

gt3d is a separate repository that uses [Plexus](https://github.com/hpcgroup/plexus) (pinned commit `c3f2660`, main,
2026-07-20; on PYTHONPATH from `/pscratch/sd/c/cunyang/gnn/plexus`) and AxoNN 0.2.0 (installed in `plexus_env` from
`/pscratch/sd/c/cunyang/gnn/axonn`) as libraries.  Nothing in either repository is modified; `tests/test_plexus_api.py`
checks the imported symbols and signatures and the pinned commit.

| gt3d | uses |
|---|---|
| `dist/grid2d.py::Grid2D` | `axonn.init(G_intra_r, G_intra_c, G_intra_d=1)` and `comm_handle.get_intra_layer_groups`: the row group is AxoNN's inner (column-parallel) group and the column group its outer (row-parallel) group for the row-major layout (swapped for column-major); `--timers` enables AxoNN's CUDA-event timers (`ax.get_timers()`, tree printed with `plexus.utils.general.print_axonn_timer_data`) |
| row collectives (`all_gather_rows`, `reduce_scatter_rows`, async start/wait, streamed layer) | `plexus.gcn_conv_rs._all_gather_rows / _reduce_scatter_rows` (wire dtype, async) with gt3d's autograd pairing |
| `--norm rms` | `plexus.scattered.ScatteredRMSNorm` |
| `--data <plexus partition dir>` | `gt3d/data/plexus_loader.py` reads `partition_graph_2d` output (metadata.pt, edge_index/0, input_features, output_labels, masks); double-permutation partitions are refused.  `tests/test_plexus_loader.py` checks it against gt3d's own partition of the same graph |

Kept in gt3d on purpose: the loss / accuracy (no per-step collectives: Plexus's `scattered_cross_entropy` adds two
small all-reduces per step, which are synchronisation points at 16 GPUs), the flat gradient all-reduce, the GCN
branch inside the fused layer (Plexus's `GCNConvRS` gathers on its own and rotates axes per layer), and the streaming
partitioner (`gt3d/data/partition.py`) for papers-scale graphs.
