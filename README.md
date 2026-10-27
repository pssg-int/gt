# gt — scalable full-graph training of sparse graph attention on many GPUs

Research code for distributed full-graph training of graph transformers with structure-masked
sparse attention (Exphormer, UniMP, Graphormer-node) on multi-GPU systems.  The implementation
targets NERSC Perlmutter (A100, Slingshot); the scripts contain site-specific paths and Slurm
accounts and need to be adapted elsewhere.

## Layout

| path | content |
|---|---|
| `gt3d/` | the system: fused Triton attention kernels, 2D / 3D process grids on AxoNN groups, fused / streamed / head-parallel layers, ring-attention baselines, sparsity-aware collectives, white-box performance and memory model, distributed trainer.  See [`gt3d/README.md`](gt3d/README.md). |
| `gt3d/bench/` | micro-benchmarks and the measured constants (`*.json`) read by the grid-selection model |
| `gt3d/tests/` | correctness tests: reference vs. backends, distributed layers vs. the single-GPU model, loaders |
| `gt3d/scripts/` | `srun` / `sbatch` wrappers for Perlmutter |
| `baseline/dac26/` | re-implementation of the DAC'26 GP-AG / GP-A2A strategies (Lin, Madduri, Kandemir; the authors' code is not public) on the paper's own stack, PyTorch 2.3 + DGL 2.3 |

Not in the repository: datasets and graph partitions, logs, profiler output, generated figures,
result logs, virtual environments, and the third-party checkouts (TorchGT, nccl-tests, Plexus,
AxoNN) the code is compared with or built on.

## Environment

Python 3.11 with the packages in [`requirements.txt`](requirements.txt).  Plexus (pinned commit
`c3f2660`, <https://github.com/hpcgroup/plexus>) and AxoNN 0.2.0 are used unmodified as libraries;
Plexus goes on `PYTHONPATH`.  The receiver-driven collectives in `gt3d/gt3d/dist/pullcomm/` need
NVSHMEM and are built with `bash gt3d/gt3d/dist/pullcomm/build.sh`.

```bash
export PYTHONPATH=/path/to/gt/gt3d:/path/to/plexus:$PYTHONPATH
export DGL_DISABLE_GRAPHBOLT=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

## Quick start

```bash
cd gt3d
python tests/test_backend_torch.py       # CPU, seconds
python tests/test_triton_kernel.py       # one GPU
python tests/test_triton_tiled.py        # one GPU
# distributed model vs. the single-GPU model (one process per GPU; torchrun here, srun in scripts/)
torchrun --nproc_per_node 4 tests/test_dist_model.py --grid 2 2 --backend torch
# training: ogbn-arxiv in memory by default, --data <gt3d partition dir> for the larger graphs
torchrun --nproc_per_node 4 gt3d/train/train_dist.py --grid 2 2 --fused
```

`gt3d/README.md` documents the layers, the grid-selection model and every command-line option.
