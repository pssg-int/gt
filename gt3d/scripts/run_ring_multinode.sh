#!/bin/bash
# Ring vs GP-AG vs 2D at 8 / 16 GPUs inside an existing interactive allocation:
#   JID=$(salloc -A gen0034_g -q interactive -N 4 -C gpu --gpus-per-node 4 -t 01:00:00 --no-shell 2>&1 | grep -o '[0-9]*$')
#   bash scripts/run_ring_multinode.sh $JID [max_nodes]
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1; MAXN=${2:-4}
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 hidden=$5 heads=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 800 srun --jobid=$JID -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid $3 $4 --backend triton --hidden $5 --heads $6 --layers 4 --timing_epochs 10 --norm ln --timers $7 2>&1 \
      | grep "^STEADY\|ring fwd\|ring bwd\|Error\|OutOfMemory" | cut -c1-220 | head -10
}
strat() {  # data nodes hidden heads extra
  nodes=$2; gp=$(( nodes * 4 ))
  case $gp in 4) g2="2 2";; 8) g2="2 4";; 16) g2="4 4";; esac
  run $1 $nodes $gp 1 $3 $4 "--ring --comm_bf16 $5"
  run $1 $nodes $gp 1 $3 $4 "--fused --comm_bf16 --rs_bf16 $5"
  run $1 $nodes $g2 $3 $4 "--fused --comm_bf16 --rs_bf16 $5"
}
for nodes in 2 4; do
  [ $nodes -gt $MAXN ] && break
  strat $P $nodes 80 2 ""
  strat $R $nodes 128 4 ""
  strat $P $nodes 256 8 "--checkpoint"
  strat $R $nodes 256 8 ""
done
echo "### done $(date +%H:%M:%S)"
