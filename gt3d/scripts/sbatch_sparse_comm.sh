#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-sparse-comm
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_sparse_comm_%j.log
# Sparsity-aware communication (gt3d/dist/sparse_comm.py), first validation beyond one node:
#   (1) bench/bench_sparse_comm.py: dense all-gather / reduce-scatter vs the sparse all-to-all-v on the real tiles
#       (SPARSE lines) and the primitive's cost curve vs the row share on each group layout (CURVE lines);
#   (2) attention-only full model (products h128 / 4 heads, LN, fused 2D, the default link): dense vs --sparse_comm
#       with the layout rule and with every collective forced sparse (--sparse_threshold 1.0);
#   (3) at 64 GPUs also papers100M (streamed layer 16x4, h64, n_sub 16): dense vs sparse sub-block gathers.
#   sbatch -q debug   -N 4  -t 00:30:00 scripts/sbatch_sparse_comm.sh     # 16 GPUs: products 4x4, 16x1
#   sbatch -q regular -N 16 -t 01:00:00 scripts/sbatch_sparse_comm.sh     # 64 GPUs: products 16x4, 8x8, 64x1; papers 16x4
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES; GP=$(( NODES * 4 ))
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; PAP=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
SRUN="srun -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores"
bench() {  # data grid_r grid_c
  echo "### bench data=$(basename $1) gpus=$GP grid=$2x$3  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_sparse_comm.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 $SRUN bash scripts/multinode_train.sh \
      --data $1 --grid $2 $3 --hidden 128 256 --heads 4 < /dev/null 2>&1 | grep "^\[bench\]\|^SPARSE\|^CURVE\|Traceback\|Error" | head -60
}
model() {  # data grid_r grid_c hidden heads extra
  echo "### model data=$(basename $1) gpus=$GP grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 $SRUN bash scripts/multinode_train.sh \
      --data $1 --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 --no_gcn $6 < /dev/null 2>&1 \
      | grep "^STEADY\|^\[sparse_comm\]\|OutOfMemory\|Traceback\|Error" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0,1,300) }' | head -4
}
case $GP in
  16) GRIDS="4 4|16 1";;
  64) GRIDS="16 4|8 8|64 1";;
  *) echo "unsupported node count $NODES"; exit 1;;
esac
IFS='|'; for g in $GRIDS; do set -- $g; unset IFS
  bench $P $1 $2
  IFS='|'
done; unset IFS
IFS='|'; for g in $GRIDS; do set -- $g; unset IFS
  model $P $1 $2 128 4 ""
  model $P $1 $2 128 4 "--sparse_comm"
  model $P $1 $2 128 4 "--sparse_comm --sparse_threshold 1.0"
  IFS='|'
done; unset IFS
if [ $GP = 64 ]; then
  # papers100M, streamed layer (the GCN branch is part of the streamed layer; same recipe as RESULTS 2026-10-15)
  pm() { echo "### model data=papers gpus=$GP grid=16x4 hidden=64 heads=4 stream_sub=16 $1  $(date +%H:%M:%S)"
    MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1500 $SRUN bash scripts/multinode_train.sh \
      --data $PAP --grid 16 4 --backend triton --hidden 64 --heads 4 --layers 4 --timing_epochs 4 --norm ln --fused --comm_bf16 --rs_bf16 --stream_sub 16 --checkpoint $1 < /dev/null 2>&1 \
      | grep "^STEADY\|^\[sparse_comm\]\|OutOfMemory\|Traceback\|Error" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0,1,300) }' | head -4; }
  bench $PAP 16 4
  pm ""
  pm "--sparse_comm --sparse_threshold 1.0"
fi
echo "### done $(date +%H:%M:%S)"
