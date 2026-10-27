#!/bin/bash
# Correctness sweep after the structure changes (int32 CSR/CSC pointers, GCNBlock transpose by sort, weights in the
# working dtype).  Run on a 4-GPU node:  srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_structure_tests.sh
cd ${GT3D_ROOT:-/pscratch/sd/c/cunyang/gnn/gt/gt3d}
export PYTHONPATH=$(pwd):/pscratch/sd/c/cunyang/gnn/plexus:$PYTHONPATH
echo "### root $(pwd)"
kernel_tests() {
echo "### single-GPU kernel tests"
for t in tests/test_backend_torch.py tests/test_triton_kernel.py tests/test_triton_tiled.py; do
  res=$(python $t 2>&1); echo "$t -> $(echo "$res" | grep -o 'ALL PASSED.*\|FAILED.*' | head -1)"; echo "$res" | grep -i 'error\|Traceback' | head -3
done
}
echo "### distributed model (fp64 vs ExphormerRef), 4 GPUs"
run() {  # label args...
  label=$1; shift
  res=$(torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "$label -> $(echo "$res" | grep -o '\[dist-model\] ALL PASSED\|\[dist-model\] FAILED' | head -1)"
  echo "$res" | grep "dist-model rank 0\|Error\|error" | grep -v ChildFailed | head -4
}
run "plain 2x2 torch"            --grid 2 2 --backend torch
run "plain 2x2 triton"           --grid 2 2 --backend triton
run "plain 4x1 triton"           --grid 4 1 --backend triton
run "plain 1x4 triton"           --grid 1 4 --backend triton
run "fused 2x2 triton"           --grid 2 2 --backend triton --fused
run "fused 1x4 triton ckpt"      --grid 1 4 --backend triton --fused --checkpoint
run "pipelined 2x2 sub2"         --grid 2 2 --backend triton --fused --pipeline_sub 2
run "streamed 2x2 sub4 torch"    --grid 2 2 --backend torch --fused --stream_sub 4
run "streamed 2x2 sub4 triton"   --grid 2 2 --backend triton --fused --stream_sub 4
run "streamed 4x1 sub4 ckpt"     --grid 4 1 --backend triton --fused --stream_sub 4 --checkpoint
run "streamed 1x4 sub2 ckpt"     --grid 1 4 --backend triton --fused --stream_sub 2 --checkpoint
run "ring 4x1 triton"            --grid 4 1 --backend triton --ring
run "ring 4x1 torch ckpt"        --grid 4 1 --backend torch --ring --checkpoint
run "fused gather_proj 2x2"      --grid 2 2 --backend triton --fused --gather_proj
run "fused gather_proj 4x1 torch" --grid 4 1 --backend torch --fused --gather_proj
run "fused gather_proj 1x4 ckpt" --grid 1 4 --backend triton --fused --gather_proj --checkpoint
echo "### distributed attention operator, 4 GPUs"
runa() { label=$1; shift
  res=$(torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_attention.py "$@" 2>&1)
  echo "$label -> $(echo "$res" | grep -o '\[dist\] ALL PASSED\|\[dist\] FAILED' | head -1)"; echo "$res" | grep "Error\|error" | grep -v ChildFailed | head -3
}
runa "attn 2x2 triton"           --grid 2 2 --backend triton
runa "attn 2x2 torch"            --grid 2 2 --backend torch
runa "attn (2,1,2) head triton"  --grid 2 1 2 --backend triton
runa "attn ring_kv 4x1 triton"   --grid 4 1 --backend triton --ring_kv
runa "attn ring_kv bidir triton" --grid 4 1 --backend triton --ring_kv --bidir
kernel_tests
echo "### done"
