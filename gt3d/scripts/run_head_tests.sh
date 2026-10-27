#!/bin/bash
# Correctness of the Grid3D head-axis attention (DistHeadSparseAttention) vs the single-process reference, 4 GPUs:
#   bash scripts/run_head_tests.sh <jobid>
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
t() {  # grid... -- extra
  GT3D_SCRIPT=tests/test_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 300 srun --jobid=$JID -N 1 --ntasks-per-node 4 --gpus-per-node 4 \
      bash scripts/multinode_train.sh "$@" 2>&1 | grep "ALL PASSED\|FAIL\|Error\|Traceback" | head -3 | sed "s|^|[$*] |"
}
for g in "2 2 1" "4 1 1" "1 4 1" "2 1 2" "1 2 2"; do
  t --grid $g --backend torch --dtype fp64
  t --grid $g --backend torch --dtype fp64 --score dot --norm softmax
  t --grid $g --backend triton --dtype fp32
  t --grid $g --backend triton --dtype fp32 --score dot --norm softmax
done
t --grid 1 1 4 --heads 4 --backend torch --dtype fp64
t --grid 1 1 4 --heads 4 --backend torch --dtype fp64 --score dot --norm softmax
t --grid 1 1 4 --heads 4 --backend triton --dtype fp32
t --grid 2 1 2 --heads 4 --backend triton --dtype fp32 --dim 32
t --grid 2 2 --backend torch --dtype fp64          # unchanged 2D path still passes
t --grid 2 2 --backend triton --dtype fp32
echo "### tests done $(date +%H:%M:%S)"
