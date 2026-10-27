#!/bin/bash
# Head grid in the full attention-only model, follow-ups on the 4-node interactive allocation:
#  (a) arxiv accuracy, official recipe (BN, 600 epochs): head grids vs the fused 2D layer, same seed;
#  (b) Reddit h128 at 4 GPUs; (c) products h80 (2 heads) and more Reddit shapes at 16 GPUs.   bash scripts/run_head_model_extra.sh <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
acc() {  # label args...
  local label=$1; shift
  echo "### acc $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1500 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --backend triton --comm_bf16 --rs_bf16 --no_gcn --epochs 600 --hidden 80 --heads 2 --layers 4 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-400 | head -3
}
run() {  # label nodes data hidden heads args...
  local label=$1 nodes=$2 data=$3 hid=$4 heads=$5; shift 5
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --no_gcn --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|head: \|qkv proj\|attention partial\|reduce-scatter\|gather x\|bwd \|'forward\|'backward\|allreduce\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce\|^|--\[95m 'head" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
acc "arxiv attn-only head (2,1,2) seed 0" --grid 2 1 2 --seed 0
acc "arxiv attn-only head (1,2,2) seed 0" --grid 1 2 2 --seed 0
acc "arxiv attn-only fused 2x2 seed 0"    --grid 2 2 --fused --seed 0
run "reddit h128 4gpu head (1,1,4)" 1 $R 128 4 --grid 1 1 4
run "reddit h128 4gpu head (2,1,2)" 1 $R 128 4 --grid 2 1 2
run "reddit h128 4gpu fused 2x2"    1 $R 128 4 --grid 2 2 --fused
for g in "2 4 2" "4 2 2" "8 1 2" "4 4 1"; do
  run "products h80 16gpu head ($g)" 4 $P 80 2 --grid $g
done
run "products h80 16gpu fused 4x4" 4 $P 80 2 --grid 4 4 --fused
run "products h80 16gpu fused 8x2" 4 $P 80 2 --grid 8 2 --fused
run "reddit h128 16gpu head (2,4,2)" 4 $R 128 4 --grid 2 4 2
run "reddit h128 16gpu head (1,4,4)" 4 $R 128 4 --grid 1 4 4
run "reddit h128 16gpu fused 8x2"    4 $R 128 4 --grid 8 2 --fused
echo "### done $(date +%H:%M:%S)"
