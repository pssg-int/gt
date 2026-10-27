#!/bin/bash
# Attention-only: bidirectional K/V ring (ring_kv2) vs the unidirectional K/V ring, inside an interactive allocation:
#   JID=$(salloc -A gen0034_g -q interactive -N 4 -C gpu --gpus-per-node 4 -t 01:30:00 --no-shell 2>&1 | grep -o '[0-9]*$')
#   bash scripts/run_attn_ringkv2.sh $JID [min_nodes] [max_nodes]
# Per GPU count and model: ring_kv2, its compute-only repeat (--no_comm), and ring_kv as the in-allocation control
# (ring / GP-AG / 2D: scripts/run_attn_ringkv.sh, RESULTS.md 2026-10-13).  bf16 communication, 10 iterations.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1; MINN=${2:-1}; MAXN=${3:-4}
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes strategy hidden heads extra grid...
  data=$1; nodes=$2; strat=$3; hid=$4; hds=$5; extra=$6; shift 6
  echo "### data=$(basename $data) nodes=$nodes gpus=$(( nodes * 4 )) strategy=$strat grid=$* hidden=$hid heads=$hds extra=$extra  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $* --strategy $strat --hidden $hid --heads $hds --iters 10 --comm_bf16 $extra < /dev/null 2>&1 \
      | grep "^ATTN\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-300 | head -4
}
study() {  # data nodes hidden heads
  nodes=$2; gp=$(( nodes * 4 ))
  run $1 $nodes ring_kv2 $3 $4 "" $gp 1
  run $1 $nodes ring_kv2 $3 $4 "--no_comm" $gp 1
  run $1 $nodes ring_kv  $3 $4 "" $gp 1
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
