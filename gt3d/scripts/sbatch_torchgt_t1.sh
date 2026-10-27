#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:40:00
#SBATCH -A m2404
#SBATCH -J gt3d-torchgt-t1
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_torchgt_t1_%j.log
# Table T1 (same model / same batch as TorchGT) at 8 and 16 GPUs: TorchGT's Graphormer-node on its arxiv 64K and
# products 870K batches, gt3d 2D grids, bf16 gathers, baseline vs fused blocks.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
export GT3D_SCRIPT=bench/torchgt_batch_bench.py
run() {  # nodes dataset seq_len split grid_r grid_c extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) $2 grid=$5x$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --dataset $2 --seq_len $3 --split $4 --grid $5 $6 --iters 20 --comm_bf16 $7 2>&1 | grep "RESULT\|Error\|OutOfMemory\|Traceback" | cut -c1-300 | head -3
}
for ex in "" "--fused"; do
  run 2 arxiv 64000 ogb 4 2 "$ex";  run 2 arxiv 64000 ogb 2 4 "$ex";  run 4 arxiv 64000 ogb 4 4 "$ex"
  run 2 products 870000 random 4 2 "$ex";  run 2 products 870000 random 2 4 "$ex"
  run 4 products 870000 random 4 4 "$ex";  run 4 products 870000 random 8 2 "$ex"
done
echo "### done $(date +%H:%M:%S)"
