#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-sync-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_sync_64_%j.log
# 64 GPUs with the pre-collective stream sync (default since 2026-10-17): the shapes the one-cross-node-axis rule
# excluded, attention operator (products h256 / 8 heads, Reddit h256) and the full model (products h128, 16x4 vs 8x8).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
attn() {  # label data hidden heads strategy grid...
  local label=$1 data=$2 hid=$3 heads=$4 strat=$5; shift 5
  echo "### attn $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --hidden $hid --heads $heads --comm_bf16 --rs_bf16 --iters 8 --strategy $strat --grid "$@" < /dev/null 2>&1 \
      | grep "^ATTN\|Traceback\|Error" | grep -v warn | cut -c1-200
}
for shape in "head 4 4 4" "head 8 2 4" "head 8 4 2" "head 16 1 4" "head 16 2 2" "head 8 1 8" "gather 16 4" "gather 8 8" "gather 32 2" "gather 64 1"; do
  set -- $shape; attn "products h256 $shape" $P 256 8 $1 ${@:2}
done
for shape in "head 4 4 4" "head 16 1 4" "gather 16 4" "gather 8 8"; do
  set -- $shape; attn "reddit h256 $shape" $R 256 8 $1 ${@:2}
done
for shape in "head 8 4 2" "head 16 2 2" "gather 16 4" "gather 8 8"; do
  set -- $shape; attn "products h80 $shape" $P 80 2 $1 ${@:2}
done
run() {  # label data pr pc hidden heads extra
  local label=$1 data=$2 pr=$3 pc=$4 hid=$5 heads=$6; shift 6
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --grid $pr $pc --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --fused --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|gather x\|reduce-scatter\|bwd \|'forward\|'backward\|allreduce\|\[cpu\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce" | cut -c1-150
}
run "products h128 16x4" $P 16 4 128 4
run "products h128 8x8"  $P 8 8 128 4
run "products h256 16x4" $P 16 4 256 8
run "products h80 16x4"  $P 16 4 80 2
run "reddit h128 16x4"   $R 16 4 128 4
run "reddit h128 8x8"    $R 8 8 128 4
echo "### done $(date +%H:%M:%S)"
