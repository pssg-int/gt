#!/bin/bash
# wait for the kernel-constants step to release its node, then: NCCL layouts on 4 nodes, head-slice kernel constants on one node
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q 'written' logs/kernel_constants_$JOBID.log; do sleep 10; done
echo "### nccl layouts $(date +%H:%M:%S)"
HW_ONLY=nccl MASTER_PORT=29660 GT3D_SCRIPT=bench/hw_constants.py timeout 900 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
    bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|Traceback\|Error'
echo "### head-slice kernel constants $(date +%H:%M:%S)"
timeout 1500 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/kernel_constants.py --configs 1x32,2x32,1x40,2x20 --regimes per_row --tag _headslices 2>&1 | grep '^\[kernel\]\|chain of\|Traceback\|Error'
echo "### done $(date +%H:%M:%S)"
