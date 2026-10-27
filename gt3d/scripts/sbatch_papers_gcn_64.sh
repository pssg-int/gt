#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-papers64g
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_papers_gcn_64_%j.log
# papers100M, 64 GPUs: GCN DCSR SpMM + compact attention kernels (both default) and TF32 matmuls.  Timing only, 5 epochs.
# TF32 is the default since 2026-10-16: the plain runs below now equal the "--tf32_matmul" ones; use --no_tf32_matmul for IEEE.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
run() {  # grid_r grid_c hidden heads stream_sub extra
  echo "### nodes=$SLURM_JOB_NUM_NODES gpus=$(( SLURM_JOB_NUM_NODES * 4 )) grid=$1x$2 hidden=$3 heads=$4 stream=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden $3 --heads $4 --layers 4 --timing_epochs 5 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $5 --timers $6 < /dev/null 2>&1 \
      | grep "^data:\|^\[compact\]\|^\[mem\]\|^STEADY\|^\[timers\]\|Max Time\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-200 | head -80
}
# GCN DCSR SpMM on top of the compact attention kernels (job 59808788: 16x4 h64 n16 4392 ms, 32x2 h128 n64 12361 ms,
# "GCN + K/V projections" 408 / 2050 ms of the forward), and TF32 matmuls for the projection / FFN GEMMs
run 16 4 64 4 16 ""
run 16 4 64 4 16 "--tf32_matmul"
run 32 2 128 4 64 ""
run 32 2 128 4 64 "--tf32_matmul"
run 16 4 64 4 8 ""
run 64 1 128 4 64 ""
echo "### done $(date +%H:%M:%S)"
