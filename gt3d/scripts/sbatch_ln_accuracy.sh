#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 00:45:00
#SBATCH -A m2404
#SBATCH -J gt3d-ln-acc
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_ln_accuracy_%j.log
# Accuracy with LayerNorm instead of BatchNorm (no statistics collectives): Reddit h128 and products h128, 4 GPUs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data hidden heads epochs norm extra
  echo "### data=$(basename $1) 4 GPUs 2x2 hidden=$2 norm=$5 epochs=$4 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1500 srun -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid 2 2 --backend triton --hidden $2 --heads $3 --layers 4 --epochs $4 --eval_every 10 --norm $5 --fused --comm_bf16 --rs_bf16 $6 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory" | cut -c1-400 | head -3
}
run $R 128 4 400 ln ""
run $P 128 4 300 ln "--checkpoint"
run $P 128 4 300 bn "--checkpoint"
echo "### done $(date +%H:%M:%S)"
