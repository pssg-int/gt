#!/bin/bash
# full model at 16 GPUs with / without the host sync before collectives (GT3D_SYNC_COLLECTIVES): does it remove the
# backward collective waiting (bwd all-gather (row) 8x nominal at 4x4)?   bash scripts/run_sync_fullmodel.sh <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label sync data pr pc hidden heads extra
  local label=$1 sync=$2 data=$3 pr=$4 pc=$5 hid=$6 heads=$7; shift 7
  echo "### $label (SYNC=$sync) $(date +%H:%M:%S)"
  GT3D_SYNC_COLLECTIVES=$sync MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $pr $pc --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "STEADY\|gather x\|reduce-scatter\|bwd \|'forward\|'backward\|allreduce\|\[cpu\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|^|--\[95m 'gather\|^|--\[95m 'reduce" | cut -c1-150
}
for sync in 0 1; do
  run "products 4x4 h128" $sync $P 4 4 128 4
  run "products 4x4 h80"  $sync $P 4 4 80 2
  run "reddit 4x4 h128"   $sync $R 4 4 128 4
  run "products 2x8 h128" $sync $P 2 8 128 4
done
run "products 4x4 h256 no-ckpt" 1 $P 4 4 256 8
run "products 4x4 h128 stream_sub 4 overlap (async path untouched)" 1 $P 4 4 128 4 --stream_sub 4 --checkpoint
echo "### done $(date +%H:%M:%S)"
