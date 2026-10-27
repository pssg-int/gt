#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:55:00
#SBATCH -A m2404
#SBATCH -J gt3d-final-ln
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_final_ln_%j.log
# Final strong-scaling tables with LayerNorm (no statistics collectives): products (hidden 80/128/256) and Reddit
# (hidden 128/256), 4 / 8 / 16 GPUs, best 2D grid vs 1D, fused layer + bf16 gathers + bf16 RS, 10 timing epochs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 hidden=$5 heads=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 700 srun -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid $3 $4 --backend triton --hidden $5 --heads $6 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $7 2>&1 \
      | grep "^STEADY\|Error\|OutOfMemory" | cut -c1-300 | head -3
}
for cfg in "80 2" "128 4 --checkpoint" "256 8 --checkpoint"; do set -- $cfg; hd=$1; hh=$2; shift 2; ex="$*"
  run $P 1 2 2 $hd $hh "$ex";   run $P 1 4 1 $hd $hh "$ex"
  run $P 2 2 4 $hd $hh "$ex";   run $P 2 8 1 $hd $hh "$ex"
  run $P 4 4 4 $hd $hh "$ex";   run $P 4 16 1 $hd $hh "$ex"
done
for cfg in "128 4" "256 8"; do set -- $cfg; hd=$1; hh=$2
  run $R 1 2 2 $hd $hh "";   run $R 1 4 1 $hd $hh ""
  run $R 2 2 4 $hd $hh "";   run $R 2 8 1 $hd $hh ""
  run $R 4 4 4 $hd $hh "";   run $R 4 16 1 $hd $hh ""
done
echo "### done $(date +%H:%M:%S)"
