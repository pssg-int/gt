#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-head-gcn-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_head_gcn_64_%j.log
# 64 GPUs: GCN + attention Exphormer (main-table model) on the head grid vs the fused 2D layer (both with --nccl_tune
# auto, the default), products h80 / h128 / h256 and Reddit h128; the attention-only head layer with the merged path.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads args...
  local label=$1 data=$2 hid=$3 heads=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[nccl_tune\]\|^\[mem\] after backward (epoch 2)\|head: \|gcn spmm\|attention partial\|reduce-scatter\|gather x\|bwd \|'forward\|'backward\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce\|^|--\[95m 'head\|^|--\[95m 'gcn\|^|--\[95m 'attention" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
for g in "4 4 4" "8 2 4" "4 8 2" "2 8 4" "8 4 2"; do
  run "products h128 GCN head ($g)" $P 128 4 --grid $g
done
run "products h128 GCN fused 8x8"  $P 128 4 --grid 8 8 --fused
run "products h128 GCN fused 16x4" $P 128 4 --grid 16 4 --fused
for g in "4 2 8" "4 4 4"; do
  run "products h256 GCN head ($g)" $P 256 8 --grid $g
done
run "products h256 GCN fused 8x8"  $P 256 8 --grid 8 8 --fused
for g in "4 8 2" "8 4 2"; do
  run "products h80 GCN head ($g)" $P 80 2 --grid $g
done
run "products h80 GCN fused 8x8"   $P 80 2 --grid 8 8 --fused
run "reddit h128 GCN head (4,4,4)" $R 128 4 --grid 4 4 4
run "reddit h128 GCN fused 8x8"    $R 128 4 --grid 8 8 --fused
run "products h128 attn-only head (4,4,4) merged" $P 128 4 --grid 4 4 4 --no_gcn
run "products h128 GCN --head_auto" $P 128 4 --head_auto
echo "### done $(date +%H:%M:%S)"
