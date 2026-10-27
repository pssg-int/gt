#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-papers32g
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_papers_gcn_32_%j.log
# papers100M on 32 GPUs (8 nodes, debug queue): GCN DCSR SpMM (2026-10-15 p.m.) on top of the compact attention
# kernels (16x2 h64 n_sub 32: 7859 ms / 32.4 GB this morning); TF32 matmuls.  Timing only, 5 epochs.
# TF32 is the default since 2026-10-16: the plain runs below now equal the "--tf32_matmul" ones; use --no_tf32_matmul for IEEE.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
run() {  # grid_r grid_c hidden heads stream_sub extra
  echo "### nodes=$SLURM_JOB_NUM_NODES gpus=$(( SLURM_JOB_NUM_NODES * 4 )) grid=$1x$2 hidden=$3 heads=$4 stream=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden $3 --heads $4 --layers 4 --timing_epochs 5 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $5 --timers $6 < /dev/null 2>&1 \
      | grep "^data:\|^\[compact\]\|^\[mem\]\|^STEADY\|Max Time\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-200 | head -60
}
# GCN DCSR SpMM + compact attention (default) vs none; TF32 matmuls; 16x2 is the only grid that fits at 32 GPUs
run 16 2 64 4 32 ""
run 16 2 64 4 32 "--no_stream_compact"
run 16 2 64 4 32 "--tf32_matmul"
run 16 2 64 4 64 ""
echo "### done $(date +%H:%M:%S)"
