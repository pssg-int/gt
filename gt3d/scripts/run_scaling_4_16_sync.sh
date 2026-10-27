#!/bin/bash
# Refreshed 4 / 8 / 16 GPU tables with the default chain (TF32, 4 heads per program, per-pass dispatch, pre-collective
# sync): products h80 / h128 / h256, Reddit h128 / h256, the 2D grids and the 1D (P,1) layout.  bash <script> <jobid>
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # nodes data pr pc hidden heads extra
  local nodes=$1 data=$2 pr=$3 pc=$4 hid=$5 heads=$6; shift 6
  echo "### $(basename $data) ${pr}x${pc} h$hid $* $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $pr $pc --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|OutOfMemory\|Traceback\|error" | grep -v warn | cut -c1-200
}
for cfg in "$P 80 2" "$P 128 4" "$P 256 8" "$R 128 4" "$R 256 8"; do
  set -- $cfg; data=$1; hid=$2; heads=$3
  ck=""; [ "$hid" = 256 ] && [ "$data" = "$P" ] && ck="--auto_ckpt"
  run 1 $data 2 2 $hid $heads $ck;  run 1 $data 4 1 $hid $heads $ck;  run 1 $data 1 4 $hid $heads $ck
  run 2 $data 2 4 $hid $heads $ck;  run 2 $data 4 2 $hid $heads $ck;  run 2 $data 8 1 $hid $heads $ck
  run 4 $data 4 4 $hid $heads;      run 4 $data 8 2 $hid $heads;      run 4 $data 2 8 $hid $heads;  run 4 $data 16 1 $hid $heads
done
echo "### done $(date +%H:%M:%S)"
