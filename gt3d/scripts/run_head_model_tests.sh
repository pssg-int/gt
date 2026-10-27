#!/bin/bash
# Correctness of the attention-only Exphormer on the head grid (Grid3D, DistExphormerLayerHead) vs the single-process
# reference (ExphormerRef with_gcn=False), 4 GPUs of an interactive allocation:   bash scripts/run_head_model_tests.sh <jobid>
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
t() {  # args of tests/test_dist_model.py
  GT3D_SCRIPT=tests/test_dist_model.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun --jobid=$JID -N 1 --ntasks-per-node 4 --gpus-per-node 4 \
      bash scripts/multinode_train.sh --graph_n 1024 "$@" 2>&1 | grep "dist-model rank 0\|ALL PASSED\|FAILED\|Error\|Traceback" | grep -v warn | head -4 | cut -c1-260 | sed "s|^|[$*] |"
}
for g in "2 1 2" "1 2 2" "2 2 1" "4 1 1" "1 4 1"; do
  t --grid $g --backend torch
  t --grid $g --backend triton
done
t --grid 1 1 4 --heads 4 --backend torch
t --grid 1 1 4 --heads 4 --backend triton
t --grid 2 1 2 --backend triton --kernel_bf16
t --grid 2 1 2 --backend torch --checkpoint
t --grid 2 2 --backend torch --no_gcn                  # attention-only reference path on the 2D grids
t --grid 2 2 --backend triton --no_gcn --fused
t --grid 2 2 --backend torch                           # unchanged full model still passes
t --grid 2 2 --backend triton --fused
echo "### tests done $(date +%H:%M:%S)"
