#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:20:00
#SBATCH -A m2404
#SBATCH -J gt3d-profile
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_profile_%j.log
# Time breakdown (torch.profiler at epoch 2, rank 0): kernels vs NCCL vs dense layers, products and Reddit, 4 and 16 GPUs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # nodes data grid_r grid_c hidden heads extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) data=$(basename $2) grid=$3x$4 hidden=$5 heads=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 500 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $2 --grid $3 $4 --backend triton --hidden $5 --heads $6 --layers 4 --timing_epochs 4 --fused --comm_bf16 --rs_bf16 --profile $7 2>&1 \
      | grep -v "^\[multinode\]\|Warning\|warn\|^\[mem\]\|^data:" | cut -c1-200 | head -60
}
run 1 $P 2 2 80 2 ""
run 4 $P 4 4 80 2 ""
run 4 $P 4 4 256 8 "--checkpoint"
run 4 $P 16 1 80 2 ""
run 1 $R 2 2 128 4 ""
run 4 $R 4 4 128 4 ""
echo "### done $(date +%H:%M:%S)"
