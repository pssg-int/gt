#!/bin/bash
# Streamed-layer / extra sparse-communication tests (rerun after the fp64 dq-cast fix of 2026-10-23); 4 GPUs of one node.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
PORT=29741; pass=0; fail=0
run() { name=$1; shift; PORT=$((PORT+1)); res=$(timeout 900 torchrun --nproc_per_node 4 --master_port $PORT tests/test_dist_model.py "$@" 2>&1)
  if echo "$res" | grep -q "\[dist-model\] ALL PASSED"; then pass=$((pass+1)); st=PASSED; else fail=$((fail+1)); st=FAILED; fi
  echo "### $st  $name :: $*"; echo "$res" | grep "dist-model rank 0\|sparse_comm info" | head -2
  [ "$st" = FAILED ] && echo "$res" | grep "Error\|line [0-9]" | grep -v site-packages | head -6; }
G="--graph_n 1024"
run "streamed torch dense control" --grid 2 2 --backend torch --fused --stream_sub 4 --stream_overlap $G --graph_e 12288
run "streamed torch"               --grid 2 2 --backend torch --fused --stream_sub 4 --sparse_comm $G --graph_e 12288
run "streamed torch overlap sparse graph" --grid 2 2 --backend torch --fused --stream_sub 4 --stream_overlap --sparse_comm $G --graph_e 800
run "streamed torch 4x1 sparse graph"     --grid 4 1 --backend torch --fused --stream_sub 2 --sparse_comm $G --graph_e 800
run "streamed torch ckpt sparse graph"    --grid 2 2 --backend torch --fused --stream_sub 4 --checkpoint --sparse_comm $G --graph_e 800
run "streamed torch 1x4 sparse graph"     --grid 1 4 --backend torch --fused --stream_sub 4 --stream_overlap --sparse_comm $G --graph_e 800
run "head torch (1,1,4) no_gcn sparse graph" --grid 1 1 4 --heads 4 --backend torch --no_gcn --sparse_comm $G --graph_e 800
run "streamed triton compact sparse graph" --grid 2 2 --backend triton --fused --stream_sub 4 --stream_overlap --sparse_comm $G --graph_e 800
echo "### SUMMARY passed=$pass failed=$fail"
