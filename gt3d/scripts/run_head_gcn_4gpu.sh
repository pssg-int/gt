#!/bin/bash
# GCN + attention Exphormer (the main-table model) on the head grid vs the fused 2D layer, 4 GPUs, products h128:
# epoch, timers, saved-tensor census; plus the attention-only head layer with the merged K | V path.   bash ... <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label hidden heads args...
  local label=$1 hid=$2 heads=$3; shift 3
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $P --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 6 --norm ln --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[census\]\|^\[mem\] after backward (epoch 2)\|head: \|qkv proj\|gcn spmm\|attention partial\|reduce-scatter\|gather x\|bwd \|'forward\|'backward\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce\|^|--\[95m 'head\|^|--\[95m 'gcn\|^|--\[95m 'qkv\|^|--\[95m 'attention" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
run "products h128 GCN head (1,1,4)" 128 4 --grid 1 1 4 --saved_census
run "products h128 GCN head (2,1,2)" 128 4 --grid 2 1 2
run "products h128 GCN head (2,2,1)" 128 4 --grid 2 2 1
run "products h128 GCN fused 2x2"    128 4 --grid 2 2 --fused
run "products h128 GCN fused 4x1"    128 4 --grid 4 1 --fused
run "products h128 attn-only head (1,1,4) merged" 128 4 --grid 1 1 4 --no_gcn
run "products h256 GCN head (1,1,4)" 256 8 --grid 1 1 4
run "products h256 GCN fused 2x2 ckpt" 256 8 --grid 2 2 --fused --checkpoint
echo "### done $(date +%H:%M:%S)"
