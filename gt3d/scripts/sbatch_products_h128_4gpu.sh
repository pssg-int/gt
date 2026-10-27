#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 00:12:00
#SBATCH -A m2404
#SBATCH -J gt3d-products-h128
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_products_h128_4gpu_%j.log
# products hidden 128 at 4 GPUs needs activation checkpointing (OOM without it in the final-table job): 2x2 and 1D 4x1.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for g in "2 2" "4 1"; do set -- $g
  echo "### nodes=1 gpus=4 grid=$1x$2 hidden=128 heads=4 --checkpoint  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 500 srun -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 10 --fused --comm_bf16 --rs_bf16 --checkpoint 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
done
echo "### done $(date +%H:%M:%S)"
