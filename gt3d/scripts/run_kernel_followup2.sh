#!/bin/bash
# after run_kernel_followup.sh: streamed-layer model tests with a sub-block-divisible graph (graph_n 1024), bf16 variants
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q '^### done' logs/kernel_followup_$JOBID.log; do sleep 20; done
echo "### streamed model tests (graph_n 1024) $(date +%H:%M:%S)"
for args in "--grid 2 2 --backend triton --fused --stream_sub 4 --checkpoint" "--grid 2 2 --backend triton --fused --stream_sub 4 --checkpoint --kernel_bf16" \
            "--grid 2 2 --backend triton --fused --stream_sub 8 --stream_overlap" "--grid 4 1 --backend triton --fused --stream_sub 4 --checkpoint --stream_overlap --kernel_bf16" \
            "--grid 2 2 --backend triton --fused --kernel_bf16" "--grid 2 2 --backend triton --fused --model unimp" "--grid 2 2 --backend triton --fused --model graphormer"; do
  echo "--- test_dist_model --graph_n 1024 $args"
  timeout 900 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh \
      torchrun --nproc_per_node 4 --master_port $(( 29900 + RANDOM % 90 )) tests/test_dist_model.py --graph_n 1024 $args 2>&1 | grep "dist-model rank 0\|ALL PASSED\|FAILED\|AssertionError\|Error" | grep -v warn | cut -c1-230
done
echo "### done $(date +%H:%M:%S)"
