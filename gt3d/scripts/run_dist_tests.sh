#!/bin/bash
# 4-GPU distributed attention tests (all grids / modes).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
for grid in "2 2" "4 1" "1 4"; do
  for cfg in "torch fp64 gated clampexp" "torch fp64 gated softmax" "torch fp64 dot softmax" "triton fp32 gated clampexp" "triton fp32 dot softmax"; do
    set -- $cfg
    res=$(torchrun --nproc_per_node 4 --master_port 29517 tests/test_dist_attention.py --grid $grid --backend $1 --dtype $2 --score $3 --norm $4 2>&1)
    echo "### grid=$grid backend=$1 dtype=$2 score=$3 norm=$4 -> $(echo "$res" | grep -o "\[dist\] ALL PASSED\|\[dist\] FAILED" | head -1)"
    echo "$res" | grep "FAIL\|Error" | grep -v "ChildFailedError\|FAILED$" | head -4
  done
done
