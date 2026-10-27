#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-nockpt
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_nockpt_4_16_%j.log
# products hidden 128 / 256 at 4-16 GPUs WITHOUT activation checkpointing (the LN table of 2026-09-27 checkpointed
# these at every GPU count for the 4-GPU memory; at 32 GPUs the checkpointed layer re-gathers in backward and costs
# 1.4-1.6x).  Same recipe otherwise: fused + bf16 gathers + bf16 RS, LayerNorm, 10 timing epochs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # nodes grid_r grid_c hidden heads extra
  echo "### data=products_p16b nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 420 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $6 < /dev/null 2>&1 \
      | grep "^STEADY\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
run 1 2 2 128 4 "";  run 1 4 1 128 4 ""
run 2 2 4 128 4 "";  run 2 8 1 128 4 ""
run 4 4 4 128 4 "";  run 4 16 1 128 4 ""
run 2 2 4 256 8 "";  run 2 8 1 256 8 ""
run 4 4 4 256 8 "";  run 4 16 1 256 8 ""
run 1 2 2 256 8 ""
echo "### done $(date +%H:%M:%S)"
