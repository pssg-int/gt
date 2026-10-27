#!/bin/bash
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
for grid in "2 2" "4 1" "1 4"; do
  for be in torch triton; do
    res=$(torchrun --nproc_per_node 4 --master_port 29519 tests/test_dist_model.py --grid $grid --backend $be 2>&1)
    echo "### grid=$grid backend=$be -> $(echo "$res" | grep -o "\[dist-model\] ALL PASSED\|\[dist-model\] FAILED" | head -1)"
    echo "$res" | grep "dist-model rank\|Error\|error" | grep -v ChildFailed | head -6
  done
done
