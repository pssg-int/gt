#!/bin/bash
# Cross-check of the whole-node layouts between nccl-tests and bench/hw_constants.py with the SAME NCCL library:
#   torch's bundled NCCL 2.27.3 (what every training run uses) vs the nccl/2.24.3 module (what nccl-tests was linked to).
#   bash scripts/run_nccl_crosscheck.sh <jobid>
JOBID=$1
BIN=/pscratch/sd/c/cunyang/gnn/gt/baseline/nccl-tests/build
TORCH_NCCL=/pscratch/sd/c/cunyang/gnn/plexus_env/lib/python3.11/site-packages/nvidia/nccl/lib
MOD_NCCL=/global/common/software/nersc9/nccl/2.24.3/lib
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
module load nccl/2.24.3 2>/dev/null
export NCCL_SOCKET_IFNAME=hsn NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_P2P_NET_CHUNKSIZE=1048576
export FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_OFLOW_BUF_SIZE=1073741824 FI_CXI_OFLOW_BUF_COUNT=1
unset MPICH_GPU_SUPPORT_ENABLED
run() {  # label k m test extra-env...
  local label=$1 k=$2 m=$3 t=$4; shift 4
  local g=$((k*m)); local sizes
  if [ "$t" = alltoall ]; then sizes="-b 4M -e 160M -f 4"; else sizes="-b $((g*4))M -e $((g*160))M -f 4"; fi
  echo "### $label layout ${k}x${m} $t $(date +%H:%M:%S)"
  env "$@" NCCL_DEBUG=INFO timeout 600 srun --jobid=$JOBID -N $m --ntasks-per-node $k --gpus-per-node 4 --cpu-bind=cores $BIN/${t}_perf $sizes -g 1 -c 0 -w 3 -n 10 2>&1 \
      | grep -m1 'NCCL version\|Using network\|NET/' ; 
  env "$@" timeout 600 srun --jobid=$JOBID -N $m --ntasks-per-node $k --gpus-per-node 4 --cpu-bind=cores $BIN/${t}_perf $sizes -g 1 -c 0 -w 3 -n 10 2>&1 \
      | grep '^ ' | cut -c1-120
}
echo "=== nccl-tests with the module's NCCL 2.24.3 (as linked) ==="
run "module-2.24.3" 4 4 all_gather
echo "=== nccl-tests with torch's NCCL 2.27.3 (LD_LIBRARY_PATH) ==="
for t in all_gather reduce_scatter alltoall; do run "torch-2.27.3" 4 2 $t LD_LIBRARY_PATH=$TORCH_NCCL:$LD_LIBRARY_PATH; run "torch-2.27.3" 4 4 $t LD_LIBRARY_PATH=$TORCH_NCCL:$LD_LIBRARY_PATH; done
echo "=== bench/hw_constants.py (torch.distributed) with the module's NCCL 2.24.3 via LD_PRELOAD ==="
HW_ONLY=nccl MASTER_PORT=29670 GT3D_SCRIPT=bench/hw_constants.py LD_PRELOAD=$MOD_NCCL/libnccl.so.2 NCCL_DEBUG=VERSION timeout 900 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
    bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|NCCL version\|Traceback\|Error' | cut -c1-160
mv bench/hw_constants_nccl.json bench/hw_constants_nccl_nccl2243.json 2>/dev/null
cp bench/hw_constants_nccl_nccl2273.json bench/hw_constants_nccl.json
echo "### done $(date +%H:%M:%S)"
