#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:28:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-diag-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_diag_32_%j.log
# 8-node debug-queue variant (layouts up to m = 8, 32-GPU grids 8x4 / 32x1 / 4x8).  Why is the column-group reduce-scatter 1.2-1.9x its alpha-beta cost at 64 GPUs (RESULTS 2026-10-17 evening / 10-18)?
#  (1) NCCL layouts the 4-node table never measured: m = 8 / 16 nodes (1x8, 1x16, 2x8, 2x16, 4x8, 4x16), back-to-back;
#  (2) the same table with training-like isolated calls (host sync before / after every call) and with a world barrier;
#  (3) NCCL knobs on the long layouts (algorithm, protocol, channels, buffer size, cross-NIC, libfabric defaults);
#  (4) the fused model on 16x4 / 8x8 / 64x1 under the most promising knobs (STEADY + timers).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
hw() {  # label tag layouts sizes isolated [ENV=VAL ...]
  local label=$1 tag=$2 layouts=$3 sizes=$4 iso=$5; shift 5
  echo "### hw $label  $(date +%H:%M:%S)"
  env "$@" HW_ONLY=nccl HW_TAG=$tag HW_LAYOUTS=$layouts HW_SIZES=$sizes HW_ISOLATED=$iso MASTER_PORT=$(( 29700 + RANDOM % 200 )) GT3D_SCRIPT=bench/hw_constants.py \
      timeout 900 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh < /dev/null 2>&1 | grep '^\[hw\]\|Traceback\|Error' | grep -v warn
}
ALL=1x2,1x4,1x8,2x1,2x2,2x4,2x8,4x1,4x2,4x4,4x8
LONG=1x4,1x8,2x8,4x4,4x8
hw "full table, back-to-back (as 10-17)"        _32bb   $ALL  0.25,1,4,9,16,64,160 0
hw "full table, isolated calls (training-like)" _32iso  $ALL  1,9,16,64            1
hw "long layouts, isolated + world barrier"     _32iso2 $LONG 9,64                 2
hw "NCCL_ALGO=Ring"              _32_ring    $LONG 9,64 1 NCCL_ALGO=Ring
hw "NCCL_ALGO=PAT"               _32_pat     $LONG 9,64 1 NCCL_ALGO=PAT
hw "NCCL_PROTO=Simple"           _32_simple  $LONG 9,64 1 NCCL_PROTO=Simple
hw "NCCL_MIN_NCHANNELS=16"       _32_ch16    $LONG 9,64 1 NCCL_MIN_NCHANNELS=16
hw "NCCL_BUFFSIZE=16MB"          _32_buf16   $LONG 9,64 1 NCCL_BUFFSIZE=16777216
hw "NCCL_NCHANNELS_PER_NET_PEER=8" _32_np8   $LONG 9,64 1 NCCL_NCHANNELS_PER_NET_PEER=8
hw "NCCL_CROSS_NIC=0"            _32_xnic0   $LONG 9,64 1 NCCL_CROSS_NIC=0
hw "libfabric defaults"          _32_fidef   $LONG 9,64 1 GT3D_FI_DEFAULTS=1
hw "NCCL_ALGO=Ring + BUFFSIZE 16MB + MIN_NCHANNELS 16" _32_combo $LONG 9,64 1 NCCL_ALGO=Ring NCCL_BUFFSIZE=16777216 NCCL_MIN_NCHANNELS=16
run() {  # label pr pc [ENV=VAL ...]
  local label=$1 pr=$2 pc=$3; shift 3
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $pr $pc --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 8 --norm ln --fused --comm_bf16 --rs_bf16 --timers < /dev/null 2>&1 \
      | grep -i "^STEADY\|gather x\|reduce-scatter\|bwd \|'forward\|'backward\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
for g in "8 4" "32 1" "4 8"; do
  run "products h128 $g baseline" $g
  run "products h128 $g NCCL_ALGO=Ring" $g NCCL_ALGO=Ring
  run "products h128 $g BUFFSIZE 16MB + MIN_NCHANNELS 16" $g NCCL_BUFFSIZE=16777216 NCCL_MIN_NCHANNELS=16
done
echo "### done $(date +%H:%M:%S)"
