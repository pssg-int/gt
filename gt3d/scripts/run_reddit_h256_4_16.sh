#!/bin/bash
# Attention-only Reddit h256 (8 heads) on every layout at 4 and 16 GPUs (model validation, 2026-10-26).
#   bash scripts/run_reddit_h256_4_16.sh <jobid>   (4-node allocation)
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # nodes label args...
  local nn=$1 label=$2; shift 2
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $nn --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $R --backend triton --hidden 256 --heads 8 --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --no_gcn --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|OutOfMemory\|Traceback\|error" | grep -v "warn" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
for g in "1 1 4" "2 1 2" "1 2 2"; do run 1 "4 GPUs reddit h256 3D ($g)" --grid $g; done
run 1 "4 GPUs reddit h256 2D 2x2"  --grid 2 2 --fused
run 1 "4 GPUs reddit h256 1D 4x1"  --grid 4 1 --fused
run 1 "4 GPUs reddit h256 ring 4x1" --grid 4 1 --ring
for g in "2 1 8" "1 2 8" "2 2 4" "4 1 4"; do run 4 "16 GPUs reddit h256 3D ($g)" --grid $g; done
run 4 "16 GPUs reddit h256 2D 4x4"  --grid 4 4 --fused
run 4 "16 GPUs reddit h256 2D 8x2"  --grid 8 2 --fused
run 4 "16 GPUs reddit h256 1D 16x1" --grid 16 1 --fused
run 4 "16 GPUs reddit h256 ring 16x1" --grid 16 1 --ring
echo "### done $(date +%H:%M:%S)"
