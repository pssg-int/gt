#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:15:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-nccl-tune-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_nccl_tune_32_%j.log
# Does train_dist.py --nccl_tune auto (NCCL_MIN_NCHANNELS=16 + NCCL_ALGO=Ring set from Python before the first collective)
# take effect and reproduce the Ring + 16 channel numbers?  products 8x4 / 32x1 (should tune), Reddit 8x4 (should not).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data grid tune [ENV=VAL ...]
  local label=$1 data=$2 grid=$3 tune=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --grid $grid --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --coll_stats --nccl_tune $tune < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\] bwd RS(col)\|^\[coll\] fwd AG(col)\|^\[nccl_tune\]\|channel{Lo..Hi}={0..15}\|OutOfMemory\|Traceback\|error" | grep -v warn | sort | uniq -c | sort -rn | head -8 | sed 's/^ *1 //'
}
run "products 8x4 --nccl_tune off"  $P "8 4" off
run "products 8x4 --nccl_tune auto" $P "8 4" auto
run "products 8x4 --nccl_tune auto + NCCL_DEBUG tuning (channels used?)" $P "8 4" auto NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING
run "products 32x1 --nccl_tune off"  $P "32 1" off
run "products 32x1 --nccl_tune auto" $P "32 1" auto
run "reddit 8x4 --nccl_tune off"  $R "8 4" off
run "reddit 8x4 --nccl_tune auto" $R "8 4" auto
echo "### done $(date +%H:%M:%S)"
