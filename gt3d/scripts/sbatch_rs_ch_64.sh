#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:40:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-ch-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_ch_64_%j.log
# 64 GPUs, within-job comparison of the NCCL channel count (NCCL_MIN_NCHANNELS=16 cut the in-situ column reduce-scatter
# from 1.38x to 1.10x of its micro-benchmark cost at 32 GPUs, jobs 59954320 / 59954592): default vs 16 channels vs
# Ring + 16 channels on the 64-GPU grids of the main table and of the head family, with per-call collective statistics.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads grid -- [ENV=VAL ...]  (2 numbers: fused GCN model; 3 numbers: head grid, attention-only)
  local label=$1 data=$2 hid=$3 heads=$4 grid=$5; shift 5
  local extra="--fused"; [ $(echo $grid | wc -w) = 3 ] && extra="--no_gcn"
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --grid $grid --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 6 --norm ln $extra --comm_bf16 --rs_bf16 --coll_stats < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|fwd RS(row)\|bwd RS(row)\|bwd AG(row)\|fwd AG(row)"
}
for shape in "$P 128 4 16 4" "$P 128 4 8 8" "$P 128 4 64 1" "$P 128 4 32 2" "$P 128 4 4 16" "$P 128 4 4 4 4" "$P 256 8 8 8" "$P 256 8 4 2 8" "$P 80 2 8 8" "$R 128 4 8 8" "$R 128 4 16 4"; do
  set -- $shape; data=$1; hid=$2; heads=$3; grid="${@:4}"
  run "$(basename $data) h$hid ($grid) default" $data $hid $heads "$grid"
  run "$(basename $data) h$hid ($grid) MIN_NCHANNELS=16" $data $hid $heads "$grid" NCCL_MIN_NCHANNELS=16
  run "$(basename $data) h$hid ($grid) Ring + MIN_NCHANNELS=16" $data $hid $heads "$grid" NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=16
done
echo "### done $(date +%H:%M:%S)"
