#!/bin/bash
# round 4: host sync before every collective (GT3D_SYNC_COLLECTIVES=1 / 2) on the real operator
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q '^### done' logs/dual_diag3_$JOBID.log; do sleep 15; done
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() { local label=$1; shift; echo "### $label  $(date +%H:%M:%S)"
  env MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py "$@" < /dev/null 2>&1 | grep "^ATTN\|^COLL\|Traceback\|Error" | grep -v warn | cut -c1-200; }
B="srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 8 --coll_timers"
run "(2,2,4) SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) SYNC_COLLECTIVES=2 (sync + group barrier)" GT3D_SYNC_COLLECTIVES=2 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) SYNC=1 + MIN_NCHANNELS=16" GT3D_SYNC_COLLECTIVES=1 NCCL_MIN_NCHANNELS=16 timeout 900 $B --strategy head --grid 2 2 4
run "(4,1,4) SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy head --grid 4 1 4
run "fused 4x4 SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy gather --grid 4 4
run "(2,1,8) SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy head --grid 2 1 8
run "(2,4,2) SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy head --grid 2 4 2
run "(2,4,2) baseline" timeout 900 $B --strategy head --grid 2 4 2
run "fused 2x8 SYNC_COLLECTIVES=1" GT3D_SYNC_COLLECTIVES=1 timeout 900 $B --strategy gather --grid 2 8
run "fused 2x8 baseline" timeout 900 $B --strategy gather --grid 2 8
echo "### done $(date +%H:%M:%S)"
