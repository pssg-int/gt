#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:20:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-ch-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_ch_32_%j.log
# NCCL_MIN_NCHANNELS=16 cut the fused model's 4x8 reduce-scatter from 51 to 41 ms per epoch (job 59954320).  How far does
# the channel count go (32), does it need Ring, does a bigger buffer add anything, is the head grid's all-to-all affected,
# and does CPU-side thread pressure matter (OMP_NUM_THREADS=1)?  Default first and last.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label nodes grid... -- [ENV=VAL ...]  (grid given as "pr pc" or "pd po pi"; head grids imply --no_gcn)
  local label=$1 nodes=$2 grid=$3; shift 3
  local extra="--fused"; [ $(echo $grid | wc -w) = 3 ] && extra="--no_gcn"
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $grid --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln $extra --comm_bf16 --rs_bf16 --coll_stats < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\]\|OutOfMemory\|Traceback\|error" | grep -v "warn\|fwd RS(row)\|bwd RS(row)\|bwd AG(row)"
}
for g in "32 1" "8 4" "4 2 4"; do
  run "$g default (1st)" 8 "$g"
  run "$g MIN_NCHANNELS=16" 8 "$g" NCCL_MIN_NCHANNELS=16
  run "$g Ring + MIN_NCHANNELS=16" 8 "$g" NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=16
  run "$g Ring + MIN_NCHANNELS=32" 8 "$g" NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=32
  run "$g MIN_NCHANNELS=16 + BUFFSIZE 16MB" 8 "$g" NCCL_MIN_NCHANNELS=16 NCCL_BUFFSIZE=16777216
  run "$g OMP_NUM_THREADS=1" 8 "$g" OMP_NUM_THREADS=1
  run "$g default (2nd)" 8 "$g"
done
echo "### done $(date +%H:%M:%S)"
