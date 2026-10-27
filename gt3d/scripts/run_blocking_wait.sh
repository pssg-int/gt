#!/bin/bash
# after the 4-16 GPU rerun: is TORCH_NCCL_BLOCKING_WAIT=1 (torch blocks the host on every collective) equivalent to the
# pre-collective sync?  (2,2,4) attention operator at 16 GPUs with our sync off.   bash <script> <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q '^### done' logs/scaling_4_16_sync_$JOBID.log; do sleep 20; done
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() { local label=$1; shift; echo "### $label  $(date +%H:%M:%S)"
  env MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py "$@" < /dev/null 2>&1 | grep "^ATTN\|^COLL total\|Traceback\|Error" | grep -v warn | cut -c1-200; }
B="srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 8 --coll_timers"
run "(2,2,4) SYNC=0 (baseline)" GT3D_SYNC_COLLECTIVES=0 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) SYNC=0 + TORCH_NCCL_BLOCKING_WAIT=1" GT3D_SYNC_COLLECTIVES=0 TORCH_NCCL_BLOCKING_WAIT=1 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) SYNC=0 + CUDA_VISIBLE_DEVICES=3,2,1,0" GT3D_SYNC_COLLECTIVES=0 CUDA_VISIBLE_DEVICES=3,2,1,0 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) SYNC=1 (default)" timeout 900 $B --strategy head --grid 2 2 4
echo "### done $(date +%H:%M:%S)"
