#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:15:00
#SBATCH -A m2404
#SBATCH -J gt3d-ln-timing
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_ln_timing_%j.log
# LayerNorm (no collectives) vs BatchNorm timing at 16 GPUs.

cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data hidden heads nosync extra
  echo "### data=$(basename $1) 16 GPUs 4x4 hidden=$2 norm=$4 $5  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 400 srun -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores --export=ALL bash scripts/multinode_train.sh \
      --data $1 --grid 4 4 --backend triton --hidden $2 --heads $3 --layers 4 --timing_epochs 10 --fused --comm_bf16 --rs_bf16 --norm $4 $5 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
}
run $R 128 4 bn ""
run $R 128 4 ln ""
run $P 80 2 bn ""
run $P 80 2 ln ""
run $P 256 8 bn "--checkpoint"
run $P 256 8 ln "--checkpoint"
echo "### done $(date +%H:%M:%S)"
