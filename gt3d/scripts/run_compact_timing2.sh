#!/bin/bash
# products 4 GPUs, streamed layer: compact attention + DCSR GCN SpMM on / off, and TF32 matmuls (JOBID = allocation).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label grid_r grid_c hidden stream_sub extra...
  label=$1; r=$2; c=$3; hid=$4; sub=$5; shift 5
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $r $c --backend triton --hidden $hid --heads 4 --layers 4 --timing_epochs 6 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $sub --timers "$@" < /dev/null 2>&1 \
      | grep "^data:\|^\[compact\]\|^\[mem\] after backward (epoch 2)\|^STEADY\|Max Time\|Error\|error:\|Traceback\|OutOfMemory" | grep -v "^|--.*stream fwd" | cut -c1-200 | head -40
}
run "h256 2x2 n_sub16 compact+gcn"     2 2 256 16
run "h256 2x2 n_sub16 no-compact"      2 2 256 16 --no_stream_compact
run "h256 2x2 n_sub64 compact+gcn"     2 2 256 64
run "h256 2x2 n_sub64 no-compact"      2 2 256 64 --no_stream_compact
run "h256 2x2 n_sub16 compact+gcn tf32" 2 2 256 16 --tf32_matmul
run "h80 2x2 n_sub64 compact+gcn"      2 2 80 64
run "h80 2x2 n_sub64 no-compact"       2 2 80 64 --no_stream_compact
echo "### done $(date +%H:%M:%S)"
