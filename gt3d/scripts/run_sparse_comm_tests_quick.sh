#!/bin/bash
# Quick sparse-communication regression (3 configs, sparse test graph), 4 GPUs of one node.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
PORT=29761; pass=0; fail=0
run() { name=$1; shift; PORT=$((PORT+1)); res=$(timeout 900 torchrun --nproc_per_node 4 --master_port $PORT tests/test_dist_model.py "$@" 2>&1)
  if echo "$res" | grep -q "\[dist-model\] ALL PASSED"; then pass=$((pass+1)); st=PASSED; else fail=$((fail+1)); st=FAILED; fi
  echo "### $st  $name :: $*"; echo "$res" | grep "dist-model rank 0" | head -1 | cut -c1-240
  [ "$st" = FAILED ] && echo "$res" | grep "Error\|line [0-9]" | grep -v site-packages | head -6; }
G="--graph_n 1024 --graph_e 800 --sparse_comm"
run "fused torch 2x2"    --grid 2 2 --backend torch --fused $G
run "fused triton 4x1 no_gcn" --grid 4 1 --backend triton --fused --no_gcn $G
run "streamed torch 2x2 overlap" --grid 2 2 --backend torch --fused --stream_sub 4 --stream_overlap $G
run "head torch (2,1,2) gcn" --grid 2 1 2 --backend torch $G
echo "### SUMMARY passed=$pass failed=$fail"
