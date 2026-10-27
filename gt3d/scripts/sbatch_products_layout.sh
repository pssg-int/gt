#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A m2404
#SBATCH -J gt3d-products-layout
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_products_layout_%j.log
# 16 GPUs: row_major vs col_major rank layouts for several grids (fused + bf16 gathers + bf16 RS).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # nodes grid_r grid_c hidden heads extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 700 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 $6 2>&1 \
      | grep "^STEADY\|Error\|error:\|Traceback" | cut -c1-300 | head -4
}
for lay in row_major col_major; do
  for grid in "4 4" "8 2" "2 8"; do
    set -- $grid
    run 4 $1 $2 80 2 "--fused --comm_bf16 --rs_bf16 --grid_layout $lay"
  done
  run 4 4 4 256 8 "--checkpoint --fused --comm_bf16 --rs_bf16 --grid_layout $lay"
  run 4 8 2 256 8 "--checkpoint --fused --comm_bf16 --rs_bf16 --grid_layout $lay"
done
echo "### done $(date +%H:%M:%S)"
