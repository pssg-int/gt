#!/bin/bash
# round 3: libfabric CXI defaults (GT3D_FI_DEFAULTS=1 in the launcher) for the dual micro-benchmark and the (2,2,4) operator
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while ! grep -q '^### done' logs/dual_diag2_$JOBID.log; do sleep 15; done
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
dual() { local label=$1; shift; echo "### dual: $label  $(date +%H:%M:%S)"
  env "$@" HW_ONLY=dual MASTER_PORT=$(( 29680 + RANDOM % 50 )) GT3D_SCRIPT=bench/hw_constants.py timeout 600 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\] dual\|Traceback\|Error' | cut -c1-260; }
run() { local label=$1; shift; echo "### $label  $(date +%H:%M:%S)"
  env MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py "$@" < /dev/null 2>&1 | grep "^ATTN\|^COLL total\|Traceback\|Error" | grep -v warn | cut -c1-200; }
B="srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 8 --coll_timers"
dual "libfabric CXI defaults" GT3D_FI_DEFAULTS=1
dual "libfabric defaults + NCCL_MIN_NCHANNELS=16" GT3D_FI_DEFAULTS=1 NCCL_MIN_NCHANNELS=16
run "(2,2,4) libfabric defaults" GT3D_FI_DEFAULTS=1 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) libfabric defaults + MIN_NCHANNELS=16" GT3D_FI_DEFAULTS=1 NCCL_MIN_NCHANNELS=16 timeout 900 $B --strategy head --grid 2 2 4
run "(4,1,4) libfabric defaults" GT3D_FI_DEFAULTS=1 timeout 900 $B --strategy head --grid 4 1 4
echo "### done $(date +%H:%M:%S)"
