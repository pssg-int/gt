#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-papers32c
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_papers_compact_32_%j.log
# papers100M on 32 GPUs (8 nodes, debug queue): first attempt since the structure slimming (2026-09-26: nothing fit
# at 40 GB); compact-accumulate kernels on / off.  Timing only, 5 epochs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
run() {  # grid_r grid_c hidden heads stream_sub extra
  echo "### nodes=$SLURM_JOB_NUM_NODES gpus=$(( SLURM_JOB_NUM_NODES * 4 )) grid=$1x$2 hidden=$3 heads=$4 stream=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden $3 --heads $4 --layers 4 --timing_epochs 5 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $5 $6 < /dev/null 2>&1 \
      | grep "^data:\|^\[compact\]\|^\[mem\]\|^STEADY\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-200 | head -30
}
run 8 4 64 4 32 ""
run 8 4 64 4 32 "--no_stream_compact"
run 8 4 64 4 64 ""
run 16 2 64 4 32 ""
echo "### done $(date +%H:%M:%S)"
