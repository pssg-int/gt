#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 01:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-ring
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_ring_motivation_%j.log
# Motivation study: context-parallel strategies for structure-masked sparse attention on 4 / 8 / 16 GPUs.
#   ring  = Ring Attention on a (P, 1) row partition (gt3d/dist/ring.py): P2P rotation of the input chunks, memory O(N/P)
#   gpag  = DAC'26 GP-AG = (P, 1) with the fused layer: all-gather of the whole input / K, V, memory O(N)
#   2d    = our 2D grid (best shape per P), fused layer
# LayerNorm, bf16 communication, 10 timing epochs, AxoNN timers (ring compute / wait breakdown).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 hidden=$5 heads=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 800 srun -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid $3 $4 --backend triton --hidden $5 --heads $6 --layers 4 --timing_epochs 10 --norm ln --timers $7 2>&1 \
      | grep "^STEADY\|ring fwd\|ring bwd\|Error\|OutOfMemory" | cut -c1-220 | head -10
}
strat() {  # data nodes hidden heads extra ; runs ring / gpag / best 2d for this GPU count
  nodes=$2; gp=$(( nodes * 4 ))
  case $gp in 4) g2="2 2";; 8) g2="2 4";; 16) g2="4 4";; esac
  run $1 $nodes $gp 1 $3 $4 "--ring --comm_bf16 $5"
  run $1 $nodes $gp 1 $3 $4 "--fused --comm_bf16 --rs_bf16 $5"
  run $1 $nodes $g2 $3 $4 "--fused --comm_bf16 --rs_bf16 $5"
}
for nodes in 1 2 4; do
  strat $P $nodes 80 2 ""
  strat $P $nodes 256 8 "--checkpoint"
  strat $R $nodes 128 4 ""
  strat $R $nodes 256 8 ""
done
echo "### done $(date +%H:%M:%S)"
