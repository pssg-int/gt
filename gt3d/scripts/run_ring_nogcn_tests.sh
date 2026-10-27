#!/bin/bash
# Attention-only ring layer (--ring --no_gcn): correctness against the single-process fp64 reference (ExphormerRef with_gcn=False).
#   srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_ring_nogcn_tests.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
t() {  # nproc  args...
  np=$1; shift
  res=$(timeout 900 torchrun --nproc_per_node $np --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "### test np=$np $* -> $(echo "$res" | grep -o "\[dist-model\] ALL PASSED\|\[dist-model\] FAILED" | head -1)"
  echo "$res" | grep "dist-model rank\|Error\|error\|Traceback" | grep -v ChildFailed | head -8
}
t 4 --grid 4 1 --ring --no_gcn --backend torch
t 4 --grid 4 1 --ring --no_gcn --backend triton
t 4 --grid 4 1 --ring --no_gcn --backend torch --checkpoint
t 2 --grid 2 1 --ring --no_gcn --backend torch
t 1 --grid 1 1 --ring --no_gcn --backend torch
t 4 --grid 4 1 --ring --backend torch                 # regression: the GCN + attention ring
t 4 --grid 2 2 --fused --no_gcn --backend torch       # regression: attention-only fused layer
echo "### done $(date +%H:%M:%S)"
