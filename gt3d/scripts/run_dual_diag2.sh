#!/bin/bash
# Narrowing the two-communicator penalty: dual micro-benchmark vs message size / NCCL channels / libfabric settings,
# and the real operator with more channels.  bash scripts/run_dual_diag2.sh <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
dual() {  # label env...
  local label=$1; shift
  echo "### dual: $label  $(date +%H:%M:%S)"
  env "$@" HW_ONLY=dual MASTER_PORT=$(( 29680 + RANDOM % 50 )) GT3D_SCRIPT=bench/hw_constants.py timeout 600 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\] dual\|Traceback\|Error' | cut -c1-260
}
run() {  # label env... -- grid args
  local label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  env MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/bench_dist_attention.py "$@" < /dev/null 2>&1 \
      | grep "^ATTN\|^COLL total\|Traceback\|Error" | grep -v warn | cut -c1-200
}
dual "baseline (4 channels per net peer, FI_CXI eager/overflow overrides)"
dual "NCCL_MIN_NCHANNELS=16" NCCL_MIN_NCHANNELS=16
dual "NCCL_MIN_NCHANNELS=32" NCCL_MIN_NCHANNELS=32
dual "NCCL_NCHANNELS_PER_NET_PEER=1" NCCL_NCHANNELS_PER_NET_PEER=1
dual "libfabric defaults (FI_CXI_* unset)" FI_CXI_RDZV_EAGER_SIZE= FI_CXI_RDZV_THRESHOLD= FI_CXI_RDZV_GET_MIN= FI_CXI_OFLOW_BUF_SIZE= FI_CXI_OFLOW_BUF_COUNT=
dual "NCCL_PROTO=Simple" NCCL_PROTO=Simple
dual "NCCL_BUFFSIZE=16M" NCCL_BUFFSIZE=16777216
B="srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh --data $P --hidden 256 --heads 8 --comm_bf16 --rs_bf16 --iters 8 --coll_timers"
run "(2,2,4) NCCL_MIN_NCHANNELS=32" NCCL_MIN_NCHANNELS=32 timeout 900 $B --strategy head --grid 2 2 4
run "(2,2,4) NCCL_BUFFSIZE=16M + MIN_NCHANNELS=16" NCCL_BUFFSIZE=16777216 NCCL_MIN_NCHANNELS=16 timeout 900 $B --strategy head --grid 2 2 4
run "(4,1,4) NCCL_MIN_NCHANNELS=16" NCCL_MIN_NCHANNELS=16 timeout 900 $B --strategy head --grid 4 1 4
run "fused 4x4 NCCL_MIN_NCHANNELS=16" NCCL_MIN_NCHANNELS=16 timeout 900 $B --strategy gather --grid 4 4
run "(2,2,4) libfabric defaults" FI_CXI_RDZV_EAGER_SIZE= FI_CXI_RDZV_THRESHOLD= FI_CXI_RDZV_GET_MIN= FI_CXI_OFLOW_BUF_SIZE= FI_CXI_OFLOW_BUF_COUNT= timeout 900 $B --strategy head --grid 2 2 4
echo "### done $(date +%H:%M:%S)"
