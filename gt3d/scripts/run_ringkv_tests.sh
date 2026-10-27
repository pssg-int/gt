#!/bin/bash
# K/V-rotating ring attention (gt3d/dist/ring_kv.py): correctness vs the single-process reference, attention only.
#   srun --jobid=$JID -N 1 --ntasks 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_ringkv_tests.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
t() {  # nproc  args...
  np=$1; shift
  res=$(timeout 600 torchrun --nproc_per_node $np --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_attention.py "$@" 2>&1)
  echo "### test np=$np $* -> $(echo "$res" | grep -o "\[dist\] ALL PASSED\|\[dist\] FAILED" | head -1)"
  echo "$res" | grep "dist rank\|Error\|error\|Traceback" | grep -v ChildFailed | head -8
}
t 4 --grid 4 1 --ring_kv --backend torch --dtype fp64
t 4 --grid 4 1 --ring_kv --backend torch --dtype fp64 --score dot
t 4 --grid 4 1 --ring_kv --backend triton --dtype fp32
t 4 --grid 4 1 --ring_kv --backend triton --dtype fp32 --num_nodes 20000 --num_edges 400000
t 2 --grid 2 1 --ring_kv --backend torch --dtype fp64
t 1 --grid 1 1 --ring_kv --backend torch --dtype fp64
t 4 --grid 4 1 --backend torch --dtype fp64            # the all-gather (P, 1) module, same harness, as a control
echo "### done $(date +%H:%M:%S)"
