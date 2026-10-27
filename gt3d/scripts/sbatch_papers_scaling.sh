#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 01:30:00
#SBATCH -A m2404
#SBATCH -J gt3d-papers100M
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_papers_scaling_%j.log
# ogbn-papers100M full-graph Exphormer on 64 GPUs: streamed layer (memory-feasible grids from grid_select), bf16 gathers
# + bf16 reduce-scatter, owned-row tail checkpoint.  Predictions (perf + memory model): see RESULTS.md 2026-09-26.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
run() {  # nodes grid_r grid_c hidden heads stream_sub extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 stream=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1500 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 6 --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $6 $7 2>&1 \
      | grep "^data:\|^\[mem\]\|^STEADY\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-300 | head -14
}
run 16 16 4 64 4 16 ""
run 16 8 8 64 4 16 ""
run 16 32 2 64 4 16 ""
run 16 64 1 64 4 32 ""
run 16 32 2 128 4 64 ""
run 16 64 1 128 4 64 ""
echo "### done $(date +%H:%M:%S)"
