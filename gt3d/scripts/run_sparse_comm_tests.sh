#!/bin/bash
# Correctness of the sparsity-aware collectives (gt3d/dist/sparse_comm.py): tests/test_dist_model.py with --sparse_comm
# (every collective forced sparse, --sparse_threshold 1.0) against the fp64 reference, on 4 GPUs of one node:
#   fused 2D layer (2x2 / 4x1 / 1x4; with and without the GCN branch), streamed layer (2x2, n_sub 4, overlap / compact),
#   head grid ((2,1,2) / (1,2,2) / (2,2,1); attention-only and with GCN), torch fp64 and triton fp32.
#   srun --jobid=<alloc> -N1 -n1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_sparse_comm_tests.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
ROOT=${GT3D_ROOT:-/pscratch/sd/c/cunyang/gnn/gt/gt3d}
export PYTHONPATH=$ROOT:$PYTHONPATH
PORT=${PORT:-29531}
pass=0; fail=0
run() {  # name args...
  name=$1; shift
  PORT=$((PORT + 1))
  res=$(timeout 900 torchrun --nproc_per_node 4 --master_port $PORT $ROOT/tests/test_dist_model.py "$@" 2>&1)
  if echo "$res" | grep -q "\[dist-model\] ALL PASSED"; then pass=$((pass + 1)); st=PASSED; else fail=$((fail + 1)); st=FAILED; fi
  echo "### $st  $name :: $*"
  echo "$res" | grep "dist-model rank 0\|sparse_comm info\|Error\|error\|Traceback" | grep -v ChildFailed | head -4
  [ "$st" = FAILED ] && echo "$res" | grep -v "^\[dist-model" | tail -15
}
S="--sparse_comm"
# fused 2D
for grid in "2 2" "4 1" "1 4"; do
  run "fused torch"        --grid $grid --backend torch  --fused $S
  run "fused torch no_gcn" --grid $grid --backend torch  --fused --no_gcn $S
done
run "fused triton"          --grid 2 2 --backend triton --fused $S
run "fused triton no_gcn 4x1" --grid 4 1 --backend triton --fused --no_gcn $S
run "fused torch gather_proj" --grid 2 2 --backend torch --fused --gather_proj $S
# streamed (chunk must be divisible by n_sub: 1024-node graph)
run "streamed torch"            --grid 2 2 --backend torch  --fused --stream_sub 4 --graph_n 1024 --graph_e 12288 $S
run "streamed torch overlap"    --grid 2 2 --backend torch  --fused --stream_sub 4 --stream_overlap --graph_n 1024 --graph_e 12288 $S
run "streamed torch 4x1"        --grid 4 1 --backend torch  --fused --stream_sub 2 --graph_n 1024 --graph_e 12288 $S
run "streamed triton compact"   --grid 2 2 --backend triton --fused --stream_sub 4 --stream_overlap --graph_n 1024 --graph_e 12288 $S
run "streamed triton no_compact" --grid 2 2 --backend triton --fused --stream_sub 4 --no_stream_compact --graph_n 1024 --graph_e 12288 $S
run "streamed torch ckpt"       --grid 2 2 --backend torch  --fused --stream_sub 4 --checkpoint --graph_n 1024 --graph_e 12288 $S
# head grid
run "head torch (2,1,2) no_gcn" --grid 2 1 2 --backend torch --no_gcn $S
run "head torch (1,2,2) no_gcn" --grid 1 2 2 --backend torch --no_gcn $S
run "head torch (2,2,1) gcn"    --grid 2 2 1 --backend torch $S
run "head torch (2,1,2) gcn"    --grid 2 1 2 --backend torch $S
run "head triton (2,1,2) no_gcn" --grid 2 1 2 --backend triton --no_gcn $S
# dense control (unchanged code paths)
run "control fused torch 2x2 dense" --grid 2 2 --backend torch --fused
run "control streamed torch dense"  --grid 2 2 --backend torch --fused --stream_sub 4 --stream_overlap --graph_n 1024 --graph_e 12288
echo "### SUMMARY passed=$pass failed=$fail"
