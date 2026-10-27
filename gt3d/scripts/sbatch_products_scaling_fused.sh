#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:45:00
#SBATCH -A m2404
#SBATCH -J gt3d-products-scaling-fused
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_products_scaling_fused_%j.log
# Strong scaling of full-graph products training with the fused-communication layer: 4, 8, 16 GPUs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # nodes  grid_r grid_c  hidden heads  extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 $6 2>&1 \
      | grep "^\[multinode\]\|^FINAL\|^STEADY\|Error\|error:\|Traceback" | cut -c1-400 | head -6
}
for extra in "--fused" "--fused --comm_bf16"; do
  run 1 2 2 80 2 "$extra"
  run 2 4 2 80 2 "$extra"
  run 2 2 4 80 2 "$extra"
  run 4 4 4 80 2 "$extra"
done
for extra in "--checkpoint" "--checkpoint --fused" "--checkpoint --fused --comm_bf16"; do
  run 1 2 2 256 8 "$extra"
  run 2 4 2 256 8 "$extra"
  run 4 4 4 256 8 "$extra"
done
echo "### done $(date +%H:%M:%S)"
