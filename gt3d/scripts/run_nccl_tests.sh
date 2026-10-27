#!/bin/bash
# nccl-tests cross-check of bench/hw_constants.py (HW_ONLY=nccl): all_gather / reduce_scatter / alltoall busbw per
# group layout "k ranks per node x m nodes", one group at a time.  bash scripts/run_nccl_tests.sh <jobid>
JOBID=$1
BIN=/pscratch/sd/c/cunyang/gnn/gt/baseline/nccl-tests/build
module load nccl/2.24.3 2>/dev/null
export NCCL_SOCKET_IFNAME=hsn NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_P2P_NET_CHUNKSIZE=1048576
export FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_OFLOW_BUF_SIZE=1073741824 FI_CXI_OFLOW_BUF_COUNT=1
unset MPICH_GPU_SUPPORT_ENABLED        # MPI only bootstraps the ranks (no GPU buffers through MPI): no GTL needed
run() {  # k m test sizes...
  local k=$1 m=$2 t=$3; shift 3
  echo "### layout ${k}x${m} (g=$((k*m))) $t $(date +%H:%M:%S)"
  timeout 600 srun --jobid=$JOBID -N $m --ntasks-per-node $k --gpus-per-node 4 --cpu-bind=cores $BIN/${t}_perf "$@" -g 1 -c 0 -w 3 -n 10 2>&1 \
      | grep -v '^#\s*$' | grep '^#\|^ ' | grep -v 'nThread\|Using devices\|Rank' | cut -c1-150
}
for lay in "4 1" "2 1" "1 2" "1 4" "2 2" "2 4" "4 2" "4 4"; do
  set -- $lay; k=$1; m=$2; g=$((k*m))
  # all-gather / reduce-scatter: nccl-tests size = n x chunk; sweep chunk 0.25 .. 160 MB -> size 0.25 g .. 160 g MB
  run $k $m all_gather -b $((g*256))K -e $((g*160))M -f 4
  run $k $m reduce_scatter -b $((g*256))K -e $((g*160))M -f 4
  # all-to-all: nccl-tests size = per-rank buffer = chunk
  run $k $m alltoall -b 256K -e 160M -f 4
done
echo "### done $(date +%H:%M:%S)"
