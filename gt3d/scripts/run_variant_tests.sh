#!/bin/bash
# UniMP / Graphormer-node variants (gt3d/models/gt_variants.py) vs their single-process references, 4 GPUs:
#   srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_variant_tests.sh
cd ${GT3D_ROOT:-/pscratch/sd/c/cunyang/gnn/gt/gt3d}
export PYTHONPATH=$(pwd):/pscratch/sd/c/cunyang/gnn/plexus:$PYTHONPATH
run() {  # label args...
  label=$1; shift
  res=$(torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "$label -> $(echo "$res" | grep -o '\[dist-model\] ALL PASSED\|\[dist-model\] FAILED' | head -1)  $(date +%H:%M:%S)"
  echo "$res" | grep "dist-model rank 0\|Error\|error\|assert" | grep -v ChildFailed | head -6
}
echo "### variants vs references  $(date +%H:%M:%S)"
run "unimp 2x2 torch fp64"        --model unimp --grid 2 2 --backend torch --graph_n 1001
run "unimp 4x1 torch fp64"        --model unimp --grid 4 1 --backend torch --graph_n 1001
run "unimp 1x4 torch ckpt"        --model unimp --grid 1 4 --backend torch --graph_n 1001 --checkpoint
run "unimp 2x2 triton fp32"       --model unimp --grid 2 2 --backend triton --graph_n 1001
run "graphormer 2x2 torch fp64"   --model graphormer --grid 2 2 --backend torch --graph_n 1001
run "graphormer 4x1 torch fp64"   --model graphormer --grid 4 1 --backend torch --graph_n 1001
run "graphormer 1x4 torch ckpt"   --model graphormer --grid 1 4 --backend torch --graph_n 1001 --checkpoint
run "graphormer 2x2 triton fp32"  --model graphormer --grid 2 2 --backend triton --graph_n 1001
run "graphormer 2x2 triton dense" --model graphormer --grid 2 2 --backend triton --graph_n 1001 --graph_e 60000
run "exphormer fused 2x2 (regression)" --model exphormer --grid 2 2 --backend torch --fused
echo "### done $(date +%H:%M:%S)"
