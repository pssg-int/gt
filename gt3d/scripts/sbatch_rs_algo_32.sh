#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:20:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-algo-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_algo_32_%j.log
# Reduce-scatter in the full model runs at 42 GB/s on the 4x8 layout (32x1 grid) against 68 in the micro-benchmark, and
# NCCL_ALGO=Ring moves it to 50 and removes the per-rank spread on the 1-rank-per-node layouts.  Which algorithm does
# NCCL pick by default, and can the micro-benchmark reproduce the training rate (hot GPU, fresh output buffers)?
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
hw() {  # label tag layouts sizes isolated [ENV=VAL ...]
  local label=$1 tag=$2 layouts=$3 sizes=$4 iso=$5; shift 5
  echo "### hw $label  $(date +%H:%M:%S)"
  env "$@" HW_ONLY=nccl HW_TAG=$tag HW_LAYOUTS=$layouts HW_SIZES=$sizes HW_ISOLATED=$iso MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/hw_constants.py \
      timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|Traceback\|Error' | grep -v warn
}
L=1x4,1x8,4x4,4x8
hw "iso baseline"                    _32a_base   $L 9,19,64 1
hw "iso + hot GPU (60 ms matmul)"    _32a_hot    $L 9,19,64 1 HW_WARMUP_MS=60
hw "iso + fresh output buffers"      _32a_fresh  $L 9,19,64 1 HW_FRESH=1
hw "iso + hot + fresh"               _32a_hotfr  $L 9,19,64 1 HW_WARMUP_MS=60 HW_FRESH=1
hw "iso, NCCL_ALGO=reducescatter:ring,allgather:ring" _32a_rsring $L 9,19,64 1 "NCCL_ALGO=reducescatter:ring;allgather:ring"
hw "iso, NCCL_ALGO=reducescatter:pat,allgather:pat"   _32a_rspat  $L 9,19,64 1 "NCCL_ALGO=reducescatter:pat;allgather:pat"
run() {  # label pr pc [ENV=VAL ...]
  local label=$1 pr=$2 pc=$3; shift 3
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $pr $pc --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --coll_stats < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\]\|OutOfMemory\|Traceback\|error" | grep -v "warn"
}
run "32x1 NCCL_ALGO=reducescatter:ring" 32 1 "NCCL_ALGO=reducescatter:ring"
run "32x1 NCCL_ALGO=reducescatter:pat"  32 1 "NCCL_ALGO=reducescatter:pat"
run "8x4 NCCL_ALGO=reducescatter:pat"   8 4  "NCCL_ALGO=reducescatter:pat"
run "32x1 NCCL_PROTO=Simple + ALGO ring" 32 1 "NCCL_ALGO=reducescatter:ring" NCCL_PROTO=Simple
echo "### NCCL tuning tables at init (rank 0), 32x1  $(date +%H:%M:%S)"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
    --data $P --grid 32 1 --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 2 --norm ln --fused --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
    | grep "NCCL INFO" | grep -i "PAT\|Latency\|AlgBw\|Ring/\|Tree/\|NVLS\|CollNet\|Using\|channels\|nChannels\|NCCL version\|Setting\|algo" | grep "\[0\] NCCL" | sort | uniq -c | sort -rn | head -60 | cut -c1-240
echo "### done $(date +%H:%M:%S)"
