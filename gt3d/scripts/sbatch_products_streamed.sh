#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A m2404
#SBATCH -J gt3d-products-streamed
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_products_streamed_%j.log
# products 16 GPUs: fused layer vs streamed layer (n_sub 4 / 8), time and steady-state peak memory.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # grid_r grid_c hidden heads extra
  echo "### grid=$1x$2 hidden=$3 heads=$4 $5  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden $3 --heads $4 --layers 4 --timing_epochs 8 --checkpoint --fused --comm_bf16 $5 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
}
run 4 4 80 2 ""
run 4 4 80 2 "--stream_sub 4"
run 4 4 256 8 "--rs_bf16"
run 4 4 256 8 "--rs_bf16 --stream_sub 4"
run 4 4 256 8 "--rs_bf16 --stream_sub 8"
run 8 2 256 8 "--rs_bf16 --stream_sub 4"
run 2 8 256 8 "--rs_bf16 --stream_sub 4"
echo "### done $(date +%H:%M:%S)"
