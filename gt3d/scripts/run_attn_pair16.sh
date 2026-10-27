#!/bin/bash
# Paired full / no-communication attention-only runs at 16 GPUs (exposed communication = full - nocomm):
#   bash scripts/run_attn_pair16.sh <jobid>
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data grid_r grid_c strategy hidden heads extra
  echo "### data=$(basename $1) gpus=16 grid=$2x$3 strategy=$4 hidden=$5 $7  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun --jobid=$JID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $1 --grid $2 $3 --strategy $4 --hidden $5 --heads $6 --iters 10 --comm_bf16 $7 2>&1 \
      | grep "^ATTN\|Error\|OutOfMemory\|Traceback" | cut -c1-300 | head -3
}
for cfg in "$P 80 2" "$P 256 8" "$R 128 4"; do set -- $cfg; D=$1; hd=$2; hh=$3
  for nc in "" "--no_comm"; do
    run $D 16 1 ring $hd $hh "$nc"
    run $D 16 1 gather $hd $hh "--rs_bf16 $nc"
    run $D 4 4 gather $hd $hh "--rs_bf16 $nc"
  done
done
echo "### done $(date +%H:%M:%S)"
