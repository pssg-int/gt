#!/bin/bash
# Diagnosis of the two-cross-node-group penalty at 16 GPUs: per-collective timers for head (2,2,4) vs (4,1,4) vs fused
# 4x4, NCCL algorithm / protocol selection, env sweeps, and the dual-group micro-benchmark.  bash scripts/run_dual_diag.sh <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label grid... -- extra env via env vars before the call
  local label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py timeout 900 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 8 "$@" < /dev/null 2>&1 \
      | grep "^ATTN\|^COLL\|^PROF\|Traceback\|Error" | grep -v warn | cut -c1-200
}
run "head (2,2,4) coll timers" --strategy head --grid 2 2 4 --coll_timers
run "head (4,1,4) coll timers" --strategy head --grid 4 1 4 --coll_timers
run "fused 4x4 coll timers"    --strategy gather --grid 4 4 --coll_timers
run "head (2,1,8) coll timers" --strategy head --grid 2 1 8 --coll_timers
echo "### NCCL algorithm / protocol selection, head (2,2,4) $(date +%H:%M:%S)"
MASTER_PORT=29777 GT3D_SCRIPT=bench/bench_dist_attention.py NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING timeout 600 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
    bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 3 --strategy head --grid 2 2 4 < /dev/null 2>&1 \
    | grep -i "algo\|proto\|Channel 00\|Connected\|Using network\|^ATTN" | grep "nid.*\[0\]\|^ATTN" | sort | uniq -c | sort -rn | head -25 | cut -c1-200
echo "### env sweeps, head (2,2,4) $(date +%H:%M:%S)"
NCCL_NCHANNELS_PER_NET_PEER=1 run "(2,2,4) NCHANNELS_PER_NET_PEER=1" --strategy head --grid 2 2 4
NCCL_ALGO=Ring run "(2,2,4) NCCL_ALGO=Ring" --strategy head --grid 2 2 4
NCCL_PROTO=Simple run "(2,2,4) NCCL_PROTO=Simple" --strategy head --grid 2 2 4
NCCL_PROTO=LL128 run "(2,2,4) NCCL_PROTO=LL128" --strategy head --grid 2 2 4
NCCL_MIN_NCHANNELS=16 run "(2,2,4) NCCL_MIN_NCHANNELS=16" --strategy head --grid 2 2 4
run "(2,2,4) baseline again" --strategy head --grid 2 2 4
echo "### dual-group micro-benchmark $(date +%H:%M:%S)"
HW_ONLY=dual MASTER_PORT=29680 GT3D_SCRIPT=bench/hw_constants.py timeout 600 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
    bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|Traceback\|Error' | cut -c1-220
echo "### done $(date +%H:%M:%S)"
