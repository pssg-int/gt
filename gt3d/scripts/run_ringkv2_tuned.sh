#!/bin/bash
# Bidirectional K/V ring with the NCCL P2P tuning (4 channels per net peer, 1 MiB net chunks) on all models at 8 / 16 GPUs,
# plus the x ring under the same tuning as a control:   bash scripts/run_ringkv2_tuned.sh $JID
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
T="NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_P2P_NET_CHUNKSIZE=1048576"
run() {  # data nodes strategy hidden heads extra
  data=$1; nodes=$2; strat=$3; hid=$4; hds=$5; extra=$6; gp=$(( nodes * 4 ))
  echo "### data=$(basename $data) nodes=$nodes gpus=$gp strategy=$strat grid=$gp 1 hidden=$hid heads=$hds extra=$extra env=$T  $(date +%H:%M:%S)"
  env $T GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $gp 1 --strategy $strat --hidden $hid --heads $hds --iters 10 --comm_bf16 $extra < /dev/null 2>&1 \
      | grep "^ATTN\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-300 | head -4
}
for nodes in 4 2; do
  run $P $nodes ring_kv2 80 2 ""
  run $P $nodes ring_kv2 256 8 ""
  run $R $nodes ring_kv2 128 4 ""
  run $R $nodes ring_kv2 256 8 ""
done
run $P 4 ring 256 8 ""
run $P 4 ring 80 2 ""
run $P 4 ring_kv2 256 8 "--no_comm"
echo "### done $(date +%H:%M:%S)"
