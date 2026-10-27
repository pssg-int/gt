#!/bin/bash
# after scripts/run_kernel_opts.sh: distributed attention test, kernel constants with 4 heads per program and bf16 operands
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q '^### done' logs/kernel_opts_$JOBID.log; do sleep 20; done
echo "### test_dist_attention (triton fp32, 2x2 / 4x1 / 1x4) $(date +%H:%M:%S)"
for g in "2 2" "4 1" "1 4"; do
  timeout 900 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh \
      torchrun --nproc_per_node 4 --master_port $(( 29900 + RANDOM % 90 )) tests/test_dist_attention.py --grid $g --backend triton --dtype fp32 2>&1 | grep -i "pass\|fail\|Traceback\|error" | grep -v warn | cut -c1-200 | tail -3
done
echo "### kernel constants: per-row 8x32 with 4 heads per program (fp32 operands) $(date +%H:%M:%S)"
timeout 1500 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py --configs 8x32 --regimes per_row --tag _hpp4 2>&1 | grep '^\[kernel\]\|Traceback\|Error'
echo "### kernel constants: bf16 operands, per-row + compact $(date +%H:%M:%S)"
timeout 2400 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py --configs 2x40,4x32,8x32 --regimes per_row,compact --tag _bf16 --lowp 2>&1 | grep '^\[kernel\]\|Traceback\|Error'
echo "### done $(date +%H:%M:%S)"
