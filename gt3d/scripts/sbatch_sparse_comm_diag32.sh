#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-sparse-diag32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_sparse_comm_diag32_%j.log
# Why is the full model slower with the column plan at 64 GPUs (16x4: +7.6 %) although the column gather / dx
# reduce-scatter are 1.1-1.2x faster in isolation (RESULTS 2026-10-23, 64-GPU section)?  32 GPUs / 8 nodes, products
# 8x4 (column group = 8 ranks, one per node, share 0.92): dense vs column-only sparse (--sparse_threshold 1.0
# --sparse_threshold_row 0), each with --timers (per-region CUDA-event tree) and --coll_stats, under the trainer's NCCL
# tuning (--nccl_tune auto: NCCL_MIN_NCHANNELS=16 + NCCL_ALGO=Ring for cross-node column groups) and with it off; then the
# collective micro-benchmark with and without the same NCCL variables.
#   sbatch -q debug -N 8 -t 00:30:00 scripts/sbatch_sparse_comm_diag32.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES; GP=$(( NODES * 4 ))
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
SRUN="srun -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores"
model() {  # tag extra
  echo "### model products gpus=$GP grid=8x4 h128 $1 :: $2  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 $SRUN bash scripts/multinode_train.sh \
      --data $P --grid 8 4 --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 --no_gcn --timers --coll_stats $2 < /dev/null 2>&1 \
      | grep "^STEADY\|^\[sparse_comm\]\|^\[nccl_tune\]\|^\[timers\]\|^\[coll\]\|gather\|reduce-scatter\|attention\|qkv\|Max Time\|OutOfMemory\|Traceback\|Error" \
      | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0,1,200) }' | head -60
}
model "dense  nccl auto" ""
model "colsp  nccl auto" "--sparse_comm --sparse_threshold 1.0 --sparse_threshold_row 0"
model "dense  nccl off"  "--nccl_tune off"
model "colsp  nccl off"  "--nccl_tune off --sparse_comm --sparse_threshold 1.0 --sparse_threshold_row 0"
bench() {  # tag env...
  echo "### bench products gpus=$GP grid=8x4 $1  $(date +%H:%M:%S)"
  env $2 GT3D_SCRIPT=bench/bench_sparse_comm.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 $SRUN bash scripts/multinode_train.sh \
      --data $P --grid 8 4 --hidden 128 256 --heads 4 --shares 0.3 0.6 0.9 1.0 < /dev/null 2>&1 | grep "^\[bench\]\|^SPARSE\|^CURVE\|Traceback\|Error" | head -40
}
bench "nccl default" ""
bench "nccl tuned (MIN_NCHANNELS=16 ALGO=Ring)" "NCCL_MIN_NCHANNELS=16 NCCL_ALGO=Ring"
echo "### done $(date +%H:%M:%S)"
