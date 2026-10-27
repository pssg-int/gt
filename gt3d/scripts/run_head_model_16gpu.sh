#!/bin/bash
# Attention-only Exphormer (--no_gcn) on the head grid vs the fused 2D layer, 16 GPUs (4 nodes), products h128 / h256 and
# Reddit h128: epoch time and timers.   bash scripts/run_head_model_16gpu.sh <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads grid... -- extra
  local label=$1 data=$2 hid=$3 heads=$4; shift 4
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --no_gcn --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|head: \|qkv proj\|attention partial\|reduce-scatter\|gather x\|bwd \|'forward\|'backward\|allreduce\|\[cpu\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce\|^|--\[95m 'head" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
for g in "2 2 4" "4 1 4" "1 4 4" "2 4 2" "4 2 2" "4 4 1"; do
  run "products h128 head ($g)" $P 128 4 --grid $g
done
run "products h128 fused 4x4" $P 128 4 --grid 4 4 --fused
run "products h128 fused 8x2" $P 128 4 --grid 8 2 --fused
for g in "2 1 8" "1 2 8" "2 2 4" "4 1 4"; do
  run "products h256 head ($g)" $P 256 8 --grid $g
done
run "products h256 fused 4x4" $P 256 8 --grid 4 4 --fused
for g in "2 2 4" "4 1 4"; do
  run "reddit h128 head ($g)" $R 128 4 --grid $g
done
run "reddit h128 fused 4x4" $R 128 4 --grid 4 4 --fused
echo "### done $(date +%H:%M:%S)"
