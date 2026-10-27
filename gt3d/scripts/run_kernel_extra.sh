#!/bin/bash
# after the NCCL layout run: compact 8x32 + head-slice per-row kernel constants on one node
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q 'written' logs/hw_nccl_raw_$JOBID.log; do sleep 10; done
echo "### kernel constants: compact 8x32 $(date +%H:%M:%S)"
timeout 1200 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py --configs 8x32 --regimes compact --tag _compact8 2>&1 | grep '^\[kernel\]\|chain of\|Traceback\|Error'
echo "### kernel constants: head slices $(date +%H:%M:%S)"
timeout 1200 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py --configs 1x32,2x32,1x40,2x20 --regimes per_row --tag _headslices 2>&1 | grep '^\[kernel\]\|chain of\|Traceback\|Error'
echo "### done $(date +%H:%M:%S)"
