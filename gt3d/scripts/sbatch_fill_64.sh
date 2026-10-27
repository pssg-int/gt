#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 01:00:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-fill-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_fill_64_%j.log
# 64 GPUs, attention-only model (--no_gcn), default chain: the points the model-validation figure still misses —
# Reddit h256 on every layout (3D / 2D / 1D / ring), and the 1D (64,1) and ring (64,1) layouts of products h128 / h256 and
# Reddit h128.  Same flags as scripts/sbatch_head_64.sh.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads args...
  local label=$1 data=$2 hid=$3 heads=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --no_gcn --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|ring fwd\|ring bwd\|'forward\|'backward\|OutOfMemory\|Traceback\|error" | grep -v "warn" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
for g in "4 2 8" "4 4 4" "16 1 4" "8 1 8"; do
  run "reddit h256 3D ($g)" $R 256 8 --grid $g
done
run "reddit h256 2D 8x8"   $R 256 8 --grid 8 8 --fused
run "reddit h256 2D 16x4"  $R 256 8 --grid 16 4 --fused
run "reddit h256 1D 64x1"  $R 256 8 --grid 64 1 --fused
run "reddit h256 ring 64x1" $R 256 8 --grid 64 1 --ring
run "reddit h128 1D 64x1"  $R 128 4 --grid 64 1 --fused
run "reddit h128 ring 64x1" $R 128 4 --grid 64 1 --ring
run "products h128 1D 64x1"  $P 128 4 --grid 64 1 --fused
run "products h128 ring 64x1" $P 128 4 --grid 64 1 --ring
run "products h256 1D 64x1"  $P 256 8 --grid 64 1 --fused
run "products h256 ring 64x1" $P 256 8 --grid 64 1 --ring
echo "### done $(date +%H:%M:%S)"
