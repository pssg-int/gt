#!/bin/bash
# Attention-only (pure GT) head-parallel study inside an interactive allocation:
#   JID=$(salloc -A gen0034_g -q interactive -N 4 -C gpu --gpus-per-node 4 -t 04:00:00 --no-shell 2>&1 | grep -o '[0-9]*$')
#   bash scripts/run_attn_head.sh $JID [min_nodes] [max_nodes]
# Per GPU count and model: fused 2D reference (strategy gather), every Grid3D shape (P_d, P_o, P_i) with P_i | H
# (P_i = 1: projected gathers; P_i = P: GP-A2A / pure head parallelism), bf16 communication.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1; MINN=${2:-1}; MAXN=${3:-4}
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes strategy hidden heads grid...
  data=$1; nodes=$2; strat=$3; hid=$4; hds=$5; shift 5
  echo "### data=$(basename $data) nodes=$nodes gpus=$(( nodes * 4 )) strategy=$strat grid=$* hidden=$hid heads=$hds  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $* --strategy $strat --hidden $hid --heads $hds --iters 10 --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
      | grep "^ATTN\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-300 | head -4
}
grids() {  # gpus heads -> list of "Pd Po Pi" with Pi | heads
  case $1 in
    4)  all="2 2 1|4 1 1|1 4 1|2 1 2|1 2 2|1 1 4";;
    8)  all="2 4 1|4 2 1|2 2 2|4 1 2|1 4 2|2 1 4|1 2 4|1 1 8";;
    16) all="4 4 1|4 2 2|2 4 2|8 1 2|1 8 2|2 2 4|4 1 4|1 4 4|2 1 8|1 2 8";;
  esac
  echo "$all" | tr '|' '\n' | while read pd po pi; do [ $(( $2 % pi )) -eq 0 ] && echo "$pd $po $pi"; done
}
study() {  # data nodes hidden heads
  nodes=$2; gp=$(( nodes * 4 ))
  case $gp in 4) g2="2 2";; 8) g2="2 4";; 16) g2="4 4";; esac
  run $1 $nodes gather $3 $4 $g2
  grids $gp $4 | while read g; do run $1 $nodes head $3 $4 $g; done
}
for nodes in 1 2 4; do
  [ $nodes -lt $MINN ] && continue
  [ $nodes -gt $MAXN ] && break
  study $P $nodes 80 2
  study $P $nodes 256 8
  study $R $nodes 128 4
  study $R $nodes 256 8
done
echo "### done $(date +%H:%M:%S)"
