#!/bin/bash
# NCCL P2P tuning sweep for the bidirectional K/V ring at 16 GPUs (products h256) inside an interactive allocation:
#   bash scripts/run_ringkv2_nccl.sh $JID
# Prints the ATTN line (and the PROF line for --profile runs) per setting; NCCL_DEBUG_SUBSYS=ENV on the first tuned run
# confirms which variables NCCL actually reads ("set by environment").
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label strategy extra_args env...
  label=$1; strat=$2; extra=$3; shift 3
  echo "### $label strategy=$strat extra=$extra env=$*  $(date +%H:%M:%S)"
  env "$@" GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JID -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $P --grid 16 1 --strategy $strat --hidden 256 --heads 8 --iters 10 --comm_bf16 $extra < /dev/null 2>&1 \
      | grep "^ATTN\|^PROF\|set by environment\|Error\|OutOfMemory\|Traceback\|AssertionError" | sort -u | cut -c1-300 | head -12
}
run baseline-prof    ring_kv2 "--profile"
run uni-prof         ring_kv  "--profile"
run chan4            ring_kv2 "" NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV
run chunk1M          ring_kv2 "" NCCL_P2P_NET_CHUNKSIZE=1048576
run chan4-chunk1M    ring_kv2 "" NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_P2P_NET_CHUNKSIZE=1048576
run chan8-chunk1M    ring_kv2 "" NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_P2P_NET_CHUNKSIZE=1048576
run buff16M          ring_kv2 "" NCCL_BUFFSIZE=16777216
run chan8-chunk1M-prof ring_kv2 "--profile" NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_P2P_NET_CHUNKSIZE=1048576
run uni-chan8-chunk1M  ring_kv  "" NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_P2P_NET_CHUNKSIZE=1048576
echo "### done $(date +%H:%M:%S)"
