#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 01:20:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-attn-head-scale
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/attn_head_scale_%j.log
# Head-parallel study (attention only, bench/bench_dist_attention.py) at 32 / 64 GPUs, continuing the 4-16 GPU study of
# scripts/run_attn_head.sh.  Submit with  sbatch -N 8  (32 GPUs)  or  sbatch -N 16 -t 01:40:00  (64 GPUs); the shape
# lists are chosen from SLURM_JOB_NUM_NODES.
# Per model: fused 2D references (P,1) = GP-AG, (P/2,2), (P/4,4) [, (P/8,8)]; Grid3D shapes with P_i in {2, 4} and
# P_d >= P_o (the 16-GPU rule), the P_d < P_o shape (2,4,4) / (4,8,2) excluded; one cross-node P_i = 8 check for the
# 8-head models; then compute-only (--no_comm) repeats of the main shapes.  bf16 communication, 10 iterations.
# Parse:  python bench/parse_attn_head.py logs/attn_head_scale_<jobid>.log
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES; GP=$(( NODES * 4 ))
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data strategy hidden heads extra grid...
  data=$1; strat=$2; hid=$3; hds=$4; extra=$5; shift 5
  echo "### data=$(basename $data) nodes=$NODES gpus=$GP strategy=$strat grid=$* hidden=$hid heads=$hds extra=$extra  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1200 srun -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $* --strategy $strat --hidden $hid --heads $hds --iters 10 --comm_bf16 --rs_bf16 $extra < /dev/null 2>&1 \
      | grep "^ATTN\|OutOfMemory\|Traceback\|AssertionError\|Error" | cut -c1-300 | head -4
}
case $GP in
  32) REF="32 1|16 2|8 4";            H2="16 1 2|8 2 2|4 4 2";          H4="8 1 4|4 2 4";          H8="4 1 8";;
  64) REF="64 1|32 2|16 4|8 8";       H2="32 1 2|16 2 2|8 4 2";         H4="16 1 4|8 2 4|4 4 4";   H8="8 1 8";;
  *) echo "unsupported node count $NODES"; exit 1;;
esac
each() { echo "$1" | tr '|' '\n'; }
study() {  # data hidden heads
  each "$REF" | while read g; do run $1 gather $2 $3 "" $g; done
  each "$H2"  | while read g; do run $1 head $2 $3 "" $g; done
  [ $3 -ge 4 ] && each "$H4" | while read g; do run $1 head $2 $3 "" $g; done
  [ $3 -ge 8 ] && each "$H8" | while read g; do run $1 head $2 $3 "" $g; done
}
# SUBSET=products | reddit | all (default): lets the 32-GPU study fit two 30-minute debug-QOS jobs.  Order: the
# 8-head products model (the decisive case) with its compute-only repeats first, so a time-truncated job keeps it.
SUBSET=${SUBSET:-all}; NC="--no_comm"
if [ "$SUBSET" != "reddit" ]; then
  study $P 256 8
  case $GP in   # compute-only repeats (exposed communication = total - compute)
    32) run $P gather 256 8 $NC 8 4;  run $P head 256 8 $NC 8 1 4;  run $P head 256 8 $NC 4 2 4;  run $P head 256 8 $NC 16 1 2;;
    64) run $P gather 256 8 $NC 8 8;  run $P gather 256 8 $NC 16 4;  run $P head 256 8 $NC 16 1 4;  run $P head 256 8 $NC 8 2 4;  run $P head 256 8 $NC 4 4 4;;
  esac
  study $P 80 2
  case $GP in
    32) run $P gather 80 2 $NC 8 4;   run $P head 80 2 $NC 16 1 2;   run $P head 80 2 $NC 8 2 2;;
    64) run $P gather 80 2 $NC 16 4;  run $P head 80 2 $NC 32 1 2;   run $P head 80 2 $NC 16 2 2;   run $P head 80 2 $NC 8 4 2;;
  esac
fi
if [ "$SUBSET" != "products" ]; then
  study $R 256 8
  case $GP in
    32) run $R gather 256 8 $NC 8 4;  run $R head 256 8 $NC 8 1 4;   run $R head 256 8 $NC 4 2 4;;
    64) run $R gather 256 8 $NC 16 4; run $R head 256 8 $NC 16 1 4;  run $R head 256 8 $NC 8 2 4;;
  esac
  study $R 128 4
fi
echo "### done $(date +%H:%M:%S)"
