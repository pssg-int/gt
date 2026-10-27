#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 01:00:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-head-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_head_64_%j.log
# 64 GPUs, default chain (TF32, 4 heads / program, per-pass dispatch, pre-collective sync, LN, bf16 wire):
#  (1) the head grid in the FULL attention-only model (--no_gcn) vs the fused 2D layer: products h128 / h256, Reddit h128;
#  (2) the missing 64-GPU main-table points of the GCN + attention model: 1D baselines (64x1, 32x2), 4x16, and 8x8 for
#      h256 / h80, Reddit 32x2 / 64x1 (E6 prediction vs measurement comes from the same STEADY lines).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads args...
  local label=$1 data=$2 hid=$3 heads=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|head: \|qkv proj\|attention partial\|reduce-scatter\|gather x\|bwd \|'forward\|'backward\|allreduce\|\[cpu\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce\|^|--\[95m 'head" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
# (1) attention-only model: head grids vs fused 2D
for g in "4 4 4" "8 2 4" "2 8 4" "4 8 2" "16 1 4" "8 4 2"; do
  run "products h128 attn-only head ($g)" $P 128 4 --no_gcn --grid $g
done
run "products h128 attn-only fused 8x8"  $P 128 4 --no_gcn --grid 8 8 --fused
run "products h128 attn-only fused 16x4" $P 128 4 --no_gcn --grid 16 4 --fused
for g in "4 2 8" "2 4 8" "4 4 4" "8 1 8"; do
  run "products h256 attn-only head ($g)" $P 256 8 --no_gcn --grid $g
done
run "products h256 attn-only fused 8x8"  $P 256 8 --no_gcn --grid 8 8 --fused
for g in "4 4 4" "16 1 4"; do
  run "reddit h128 attn-only head ($g)" $R 128 4 --no_gcn --grid $g
done
run "reddit h128 attn-only fused 8x8"  $R 128 4 --no_gcn --grid 8 8 --fused
# (2) GCN + attention main-table points still missing at 64 GPUs
run "products h128 64x1" $P 128 4 --grid 64 1 --fused
run "products h128 32x2" $P 128 4 --grid 32 2 --fused
run "products h128 4x16" $P 128 4 --grid 4 16 --fused
run "products h256 8x8"  $P 256 8 --grid 8 8 --fused
run "products h256 32x2" $P 256 8 --grid 32 2 --fused
run "products h80 8x8"   $P 80 2 --grid 8 8 --fused
run "products h80 32x2"  $P 80 2 --grid 32 2 --fused
run "reddit h128 32x2"   $R 128 4 --grid 32 2 --fused
run "reddit h128 64x1"   $R 128 4 --grid 64 1 --fused
echo "### done $(date +%H:%M:%S)"
