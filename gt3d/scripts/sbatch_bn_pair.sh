#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:15:00
#SBATCH -A m2404
#SBATCH -J gt3d-bn-pair
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_bn_pair_%j.log
# Steady-state effect of sharing one statistics all-reduce between the two BatchNorms of a layer (GT3D_BN_PAIR=1 vs 0).

cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data hidden heads nosync extra
  echo "### data=$(basename $1) 16 GPUs 4x4 hidden=$2 bn_pair=$4 $5  $(date +%H:%M:%S)"
  GT3D_BN_PAIR=$4 MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 400 srun -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores --export=ALL bash scripts/multinode_train.sh \
      --data $1 --grid 4 4 --backend triton --hidden $2 --heads $3 --layers 4 --timing_epochs 10 --fused --comm_bf16 --rs_bf16 $5 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
}
run $R 128 4 0 ""
run $R 128 4 1 ""
run $P 80 2 0 ""
run $P 80 2 1 ""
run $P 256 8 0 "--checkpoint"
run $P 256 8 1 "--checkpoint"
echo "### done $(date +%H:%M:%S)"
