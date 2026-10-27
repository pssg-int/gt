#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:40:00
#SBATCH -A m2404
#SBATCH -J gt3d-products-rsbf16
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_products_rsbf16_%j.log
# 16 GPUs: fused + bf16 gathers, with/without bf16 reduce-scatter; alternative 16-GPU grids; plus a 100-epoch accuracy check.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # nodes grid_r grid_c hidden heads extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 700 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 $6 2>&1 \
      | grep "^STEADY\|^FINAL\|Error\|error:\|Traceback" | sed -e 's/data=[^ ]* //' | cut -c1-400 | head -4
}
T="--timing_epochs 10"
run 4 4 4 80 2 "--fused --comm_bf16 $T"
run 4 4 4 80 2 "--fused --comm_bf16 --rs_bf16 $T"
run 4 2 8 80 2 "--fused --comm_bf16 --rs_bf16 $T"
run 4 8 2 80 2 "--fused --comm_bf16 --rs_bf16 $T"
run 4 4 4 256 8 "--checkpoint --fused --comm_bf16 $T"
run 4 4 4 256 8 "--checkpoint --fused --comm_bf16 --rs_bf16 $T"
run 4 2 8 256 8 "--checkpoint --fused --comm_bf16 --rs_bf16 $T"
echo "### accuracy check: 100 epochs, 4 GPUs, fused+bf16 vs +rs_bf16"
run 1 2 2 80 2 "--fused --comm_bf16 --epochs 100 --eval_every 5"
run 1 2 2 80 2 "--fused --comm_bf16 --rs_bf16 --epochs 100 --eval_every 5"
echo "### done $(date +%H:%M:%S)"
