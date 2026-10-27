#!/bin/bash
# Attention-only ring layer (--ring --no_gcn) in the full model on 4 and 16 GPUs, the ring P2P constants, and a saved-tensor
# census of the ring layer.   bash scripts/run_ring_nogcn_timing.sh <jobid>   (4-node allocation)
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
echo "### P2P constants  $(date +%H:%M:%S)"
for nn in 1 4; do
  GT3D_SCRIPT=bench/ring_p2p.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun --jobid=$JOBID -N $nn --ntasks-per-node 4 --gpus-per-node 4 \
      bash scripts/multinode_train.sh < /dev/null 2>&1 | grep "^P2P\|Error\|Traceback"
done
run() {  # nodes label data hidden heads args...
  local nn=$1 label=$2 data=$3 hid=$4 heads=$5; shift 5
  echo "### model $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $nn --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 --no_gcn --timers "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[mem\] after backward (epoch 2)\|^\[census\]\|ring fwd\|ring bwd\|'forward\|'backward\|allreduce\|OutOfMemory\|Traceback\|error" | grep -v "warn" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0, 1, 160) }'
}
# 4 GPUs (1 node)
run 1 "4 GPUs products h128 ring (4,1)"  $P 128 4 --grid 4 1 --ring
run 1 "4 GPUs products h256 ring (4,1)"  $P 256 8 --grid 4 1 --ring
run 1 "4 GPUs reddit h128 ring (4,1)"    $R 128 4 --grid 4 1 --ring
run 1 "4 GPUs reddit h128 1D (4,1)"      $R 128 4 --grid 4 1 --fused
# 16 GPUs (4 nodes)
run 4 "16 GPUs products h128 ring (16,1)" $P 128 4 --grid 16 1 --ring
run 4 "16 GPUs products h256 ring (16,1)" $P 256 8 --grid 16 1 --ring
run 4 "16 GPUs products h80 ring (16,1)"  $P 80 2 --grid 16 1 --ring
run 4 "16 GPUs reddit h128 ring (16,1)"   $R 128 4 --grid 16 1 --ring
run 4 "16 GPUs reddit h128 1D (16,1)"     $R 128 4 --grid 16 1 --fused
# saved-tensor census of the ring layer (memory constant; separate run, the census perturbs the timing)
run 1 "census products h128 ring (4,1)"   $P 128 4 --grid 4 1 --ring --saved_census --timing_epochs 3
echo "### done $(date +%H:%M:%S)"
