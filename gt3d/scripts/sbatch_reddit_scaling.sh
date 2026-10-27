#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:35:00
#SBATCH -A m2404
#SBATCH -J gt3d-reddit-scaling
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_reddit_scaling_%j.log
# Reddit strong scaling: 4 / 8 / 16 GPUs, best 2D grid (perf model) vs 1D (P x 1), hidden 80 / 128 / 256,
# fused layer + bf16 gathers + bf16 reduce-scatter, checkpoint for hidden 256; 10 timing epochs each.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # nodes grid_r grid_c hidden heads extra
  echo "### nodes=$1 gpus=$(( $1 * 4 )) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 700 srun -N $1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 --fused --comm_bf16 --rs_bf16 $6 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
}
for cfg in "128 4" "256 8"; do set -- $cfg; hd=$1; hh=$2; shift 2; ex="$*"
  run 1 2 2 $hd $hh "$ex";   run 1 4 1 $hd $hh "$ex"
  run 2 4 2 $hd $hh "$ex";   run 2 2 4 $hd $hh "$ex";   run 2 8 1 $hd $hh "$ex"
  run 4 4 4 $hd $hh "$ex";   run 4 8 2 $hd $hh "$ex";   run 4 16 1 $hd $hh "$ex"
done
echo "### done $(date +%H:%M:%S)"
