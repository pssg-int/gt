#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:25:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-ctrl-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_ctrl_32_%j.log
# Controlled, within-job comparison (run-to-run variance across jobs is ~10 %): the fused model's column-group
# reduce-scatter under the NCCL settings, default repeated first and last; and the micro-benchmark with a properly
# synchronised hot GPU (busy loop, then world barrier, then the call) and with the input rewritten before every call.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label nodes pr pc [ENV=VAL ...]
  local label=$1 nodes=$2 pr=$3 pc=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $pr $pc --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --coll_stats < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\] bwd RS(col)\|^\[coll\] fwd AG(col)\|^\[coll\] fwd RS(row)\|OutOfMemory\|Traceback\|error" | grep -v "warn"
}
hw() {  # label tag layouts sizes isolated [ENV=VAL ...]
  local label=$1 tag=$2 layouts=$3 sizes=$4 iso=$5; shift 5
  echo "### hw $label  $(date +%H:%M:%S)"
  env "$@" HW_ONLY=nccl HW_TAG=$tag HW_LAYOUTS=$layouts HW_SIZES=$sizes HW_ISOLATED=$iso MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/hw_constants.py \
      timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|Traceback\|Error' | grep -v warn
}
for g in "32 1" "8 4"; do
  run "$g default (1st)" 8 $g
  run "$g NCCL_ALGO=Ring" 8 $g NCCL_ALGO=Ring
  run "$g NCCL_ALGO=Ring NCCL_PROTO=Simple" 8 $g NCCL_ALGO=Ring NCCL_PROTO=Simple
  run "$g reducescatter:ring;allgather:ring + Simple" 8 $g "NCCL_ALGO=reducescatter:ring;allgather:ring" NCCL_PROTO=Simple
  run "$g NCCL_PROTO=Simple only" 8 $g NCCL_PROTO=Simple
  run "$g NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=16" 8 $g NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=16
  run "$g default (2nd)" 8 $g
done
hw "iso2 baseline"                      _32c_base   1x8,4x8 19,64 2
hw "iso2 + hot GPU 200 ms then barrier" _32c_hot    1x8,4x8 19,64 2 HW_WARMUP_MS=200
hw "iso2 + input rewritten before call" _32c_touch  1x8,4x8 19,64 2 HW_TOUCH=1
hw "iso2 + hot + touch"                 _32c_hottch 1x8,4x8 19,64 2 HW_WARMUP_MS=200 HW_TOUCH=1
hw "iso2 NCCL_ALGO=Ring"                _32c_ring   1x8,4x8 19,64 2 NCCL_ALGO=Ring
echo "### done $(date +%H:%M:%S)"
