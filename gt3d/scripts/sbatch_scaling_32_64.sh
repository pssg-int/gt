#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-scale-32-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_scaling_32_64_%j.log
# Full-model strong scaling beyond 16 GPUs (continues the LayerNorm tables of scripts/sbatch_final_ln.sh with the same
# recipe: fused layer + bf16 gathers + bf16 reduce-scatter, LayerNorm, 10 timing epochs, steady state).
#   sbatch -q debug   -N 8  -t 00:30:00 scripts/sbatch_scaling_32_64.sh     # 32 GPUs
#   sbatch -q regular -N 16 -t 00:45:00 scripts/sbatch_scaling_32_64.sh     # 64 GPUs
# Grids per GPU count: the node-filling 2D grid (P/4, 4), the 1D all-gather baseline (P, 1), the intermediate (P/2, 2)
# and, at 64 GPUs, (8, 8) (row group across two nodes).  hidden 128 / 256 keep --checkpoint as in the 4-16 GPU table;
# the best 2D grid is repeated without checkpointing.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES; GP=$(( NODES * 4 ))
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) nodes=$NODES gpus=$GP grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 420 srun -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $6 < /dev/null 2>&1 \
      | grep "^STEADY\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
case $GP in
  32) G2D="8 4";  G1D="32 1"; GMID="16 2"; GEXTRA="";;
  64) G2D="16 4"; G1D="64 1"; GMID="32 2"; GEXTRA="8 8";;
  *) echo "unsupported node count $NODES"; exit 1;;
esac
# main shapes first (2D best vs 1D), every model
for cfg in "80 2" "128 4 --checkpoint" "256 8 --checkpoint"; do set -- $cfg; hd=$1; hh=$2; shift 2; ex="$*"
  run $P $G2D $hd $hh "$ex";  run $P $G1D $hd $hh "$ex"
done
for cfg in "128 4" "256 8"; do set -- $cfg; hd=$1; hh=$2
  run $R $G2D $hd $hh "";  run $R $G1D $hd $hh ""
done
# intermediate grids (grid scan for the performance model)
for cfg in "80 2" "128 4 --checkpoint" "256 8 --checkpoint"; do set -- $cfg; hd=$1; hh=$2; shift 2; ex="$*"
  run $P $GMID $hd $hh "$ex"
  [ -n "$GEXTRA" ] && run $P $GEXTRA $hd $hh "$ex"
done
for cfg in "128 4" "256 8"; do set -- $cfg; hd=$1; hh=$2
  run $R $GMID $hd $hh ""
  [ -n "$GEXTRA" ] && run $R $GEXTRA $hd $hh ""
done
# best 2D grid without checkpointing
run $P $G2D 128 4 "";  run $P $G2D 256 8 ""
echo "### done $(date +%H:%M:%S)"
