#!/bin/bash
#SBATCH -C "gpu&hbm40g"
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-pull-nvshmem
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_pull_nvshmem_%j.log
# Cross-node receiver-driven sparse collectives (NVSHMEM get by row-index table, fused reduce; gt3d/dist/pullcomm)
# vs NCCL all-gather / reduce-scatter and the a2a-v plan, on a ONE-RANK-PER-NODE group (the layout of GT3D's column
# groups): bench/bench_pull_nvshmem.py with --ntasks-per-node 1.
#   sbatch -q debug -N 4 -t 00:20:00 scripts/sbatch_pull_nvshmem.sh      # 4 PEs on 4 nodes
#   sbatch -q debug -N 8 -t 00:20:00 scripts/sbatch_pull_nvshmem.sh      # 8 PEs on 8 nodes
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES
run() {  # rows extra
  echo "### nvshmem pull: nodes=$NODES PEs=$NODES (one per node) rows/rank=$1 $2  $(date +%H:%M:%S)"
  PYTHONFAULTHANDLER=1 GT3D_SCRIPT=bench/bench_pull_nvshmem.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 \
    srun -N $NODES --ntasks-per-node 1 --gpus-per-node 1 --cpu-bind=cores bash scripts/run_nvshmem.sh --rows $1 --dims 128 256 --densities 0.2 0.45 0.8 1.0 $2 < /dev/null 2>&1 \
    | grep "^\[nvshmem\]\|^NVSHMEM \|Fatal\|  File \"/pscratch\|Segmentation\|assert\|Error\|WARN" | grep -v "site-packages" | cut -c1-500 | head -40
}
run 153088 ""
run 612352 ""
echo "### done $(date +%H:%M:%S)"
