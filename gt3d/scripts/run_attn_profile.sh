#!/bin/bash
# Attention-only benchmark with the kernel-time breakdown (NCCL share) for the sparse-communication estimate:
#   bash scripts/run_attn_profile.sh <jobid>
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes grid_r grid_c strategy hidden heads extra
  echo "### data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 strategy=$5 hidden=$6 heads=$7 $8  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun --jobid=$JID --overlap -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $1 --grid $3 $4 --strategy $5 --hidden $6 --heads $7 --iters 6 --comm_bf16 --profile $8 2>&1 \
      | grep "^ATTN\|^PROF\|Error\|OutOfMemory\|Traceback" | cut -c1-330 | head -4
}
strat() {  # data nodes hidden heads
  nodes=$2; gp=$(( nodes * 4 ))
  case $gp in 4) g2="2 2";; 8) g2="2 4";; 16) g2="4 4";; esac
  run $1 $nodes $gp 1 ring $3 $4 ""
  run $1 $nodes $gp 1 gather $3 $4 "--rs_bf16"
  run $1 $nodes $g2 gather $3 $4 "--rs_bf16"
}
strat $P 4 80 2
strat $P 4 256 8
strat $R 4 128 4
strat $P 1 80 2
echo "### done $(date +%H:%M:%S)"
