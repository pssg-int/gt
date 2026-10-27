#!/bin/bash
# products 4 GPUs, streamed layer: compact-accumulate kernels on / off (JOBID = interactive allocation).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label grid_r grid_c hidden stream_sub extra...
  label=$1; r=$2; c=$3; hid=$4; sub=$5; shift 5
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $r $c --backend triton --hidden $hid --heads 4 --layers 4 --timing_epochs 6 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $sub "$@" < /dev/null 2>&1 \
      | grep "^data:\|^\[mem\] after backward (epoch 2)\|^STEADY\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-220 | head -12
}
run "h256 2x2 n_sub16 compact"     2 2 256 16
run "h256 2x2 n_sub16 no-compact"  2 2 256 16 --no_stream_compact
run "h256 2x2 n_sub4 compact"      2 2 256 4
run "h256 2x2 n_sub4 no-compact"   2 2 256 4 --no_stream_compact
run "h80 4x1 n_sub16 compact"      4 1 80 16
run "h80 4x1 n_sub16 no-compact"   4 1 80 16 --no_stream_compact
echo "### done $(date +%H:%M:%S)"
