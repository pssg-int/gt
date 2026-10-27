#!/bin/bash
# Correctness of the compact-accumulate (DCSR) attention kernels and the streamed layer that uses them, plus the
# kernel micro-benchmark.  Run on a 4-GPU node:
#   srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_compact_tests.sh
cd ${GT3D_ROOT:-/pscratch/sd/c/cunyang/gnn/gt/gt3d}
export PYTHONPATH=$(pwd):/pscratch/sd/c/cunyang/gnn/plexus:$PYTHONPATH
echo "### root $(pwd)  $(date +%H:%M:%S)"
run() {  # label args...
  label=$1; shift
  res=$(torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "$label -> $(echo "$res" | grep -o '\[dist-model\] ALL PASSED\|\[dist-model\] FAILED' | head -1)  $(date +%H:%M:%S)"
  echo "$res" | grep "dist-model rank 0\|Error\|error" | grep -v ChildFailed | head -6
}
echo "### streamed model, triton fp32 vs torch reference, 4 GPUs (sparse sub-blocks -> compact kernels)"
run "streamed 2x2 sub4 triton"          --grid 2 2 --backend triton --fused --stream_sub 4 --graph_n 1024
run "streamed 2x2 sub4 triton no-compact" --grid 2 2 --backend triton --fused --stream_sub 4 --graph_n 1024 --no_stream_compact
run "streamed 4x1 sub4 ckpt overlap"     --grid 4 1 --backend triton --fused --stream_sub 4 --checkpoint --stream_overlap --graph_n 1024
run "streamed 1x4 sub2 ckpt"             --grid 1 4 --backend triton --fused --stream_sub 2 --checkpoint --graph_n 1024
run "streamed 2x2 sub8 ckpt dense"       --grid 2 2 --backend triton --fused --stream_sub 8 --checkpoint --graph_n 1024 --graph_e 98304
echo "### tiled kernels vs per-row kernels (incl. compact-accumulate)  $(date +%H:%M:%S)"
res=$(python tests/test_triton_tiled.py 2>&1); echo "$res" | grep -i "accum\|ALL PASSED\|FAILED\|error\|Traceback" | head -60
echo "### row-overhead micro-benchmark (3.5M rows, h4 d16)  $(date +%H:%M:%S)"
python bench/bench_row_overhead.py --block_r 16 2>&1 | grep -v Warning
echo "### done $(date +%H:%M:%S)"
