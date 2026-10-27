#!/bin/bash
# Streamed layer with the compact-accumulate kernels vs the torch reference (fp64 reference, triton fp32 path).
cd ${GT3D_ROOT:-/pscratch/sd/c/cunyang/gnn/gt/gt3d}
export PYTHONPATH=$(pwd):/pscratch/sd/c/cunyang/gnn/plexus:$PYTHONPATH
run() {  # label args...
  label=$1; shift
  res=$(torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "$label -> $(echo "$res" | grep -o '\[dist-model\] ALL PASSED\|\[dist-model\] FAILED' | head -1)  $(date +%H:%M:%S)"
  echo "$res" | grep "dist-model rank 0\|Error\|error" | grep -v ChildFailed | head -6
}
echo "### streamed model, 4 GPUs (sparse sub-blocks -> compact kernels)  $(date +%H:%M:%S)"
run "streamed 2x2 sub4 triton"            --grid 2 2 --backend triton --fused --stream_sub 4 --graph_n 1024
run "streamed 2x2 sub4 triton no-compact" --grid 2 2 --backend triton --fused --stream_sub 4 --graph_n 1024 --no_stream_compact
run "streamed 4x1 sub4 ckpt overlap"      --grid 4 1 --backend triton --fused --stream_sub 4 --checkpoint --stream_overlap --graph_n 1024
run "streamed 1x4 sub2 ckpt"              --grid 1 4 --backend triton --fused --stream_sub 2 --checkpoint --graph_n 1024
run "streamed 2x2 sub8 ckpt dense"        --grid 2 2 --backend triton --fused --stream_sub 8 --checkpoint --graph_n 1024 --graph_e 98304
run "streamed 2x2 sub4 torch"             --grid 2 2 --backend torch --fused --stream_sub 4 --graph_n 1024
echo "### done $(date +%H:%M:%S)"
# GCN DCSR SpMM (2026-10-15 p.m.): the torch fp64 run above validates it at 1e-15; reference without it
run "streamed 2x2 sub4 torch no-compact"  --grid 2 2 --backend torch --fused --stream_sub 4 --graph_n 1024 --no_stream_compact
run "streamed 4x1 sub8 torch ckpt"        --grid 4 1 --backend torch --fused --stream_sub 8 --checkpoint --graph_n 1024
echo "### done (gcn compact) $(date +%H:%M:%S)"
