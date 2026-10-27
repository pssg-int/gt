#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 8
#SBATCH --gpus-per-node 4
#SBATCH -t 00:20:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-rs-skew-32
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_rs_skew_32_%j.log
# Column-group reduce-scatter excess in the full model: per-call max / mean / MIN over ranks (--coll_stats; min = the
# collective itself, mean - min = arrival skew) for the 32-GPU grids 8x4 (column group 1x8), 32x1 (4x8), 4x8 (1x4), with
# the default NCCL algorithm and NCCL_ALGO=Ring; 16-GPU 4x4 for reference; one run with NCCL's collective log.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # label nodes pr pc [ENV=VAL ...]
  local label=$1 nodes=$2 pr=$3 pc=$4; shift 4
  echo "### model $label  $(date +%H:%M:%S)"
  env "$@" MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid $pr $pc --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --timers --coll_stats < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[coll\]\|attn bwd kernel\|attention partial\|gcn spmm\|'backward\|'forward\|OutOfMemory\|Traceback\|error" | grep -v "warn" | awk '{ if ($0 ~ /^STEADY|^\[coll\]/) print; else print substr($0, 1, 160) }'
}
for g in "8 4" "32 1" "4 8"; do
  run "products h128 $g default algo" 8 $g
  run "products h128 $g NCCL_ALGO=Ring" 8 $g NCCL_ALGO=Ring
done
run "products h128 4 4 (16 GPUs) default algo" 4 4 4
run "products h128 4 4 (16 GPUs) NCCL_ALGO=Ring" 4 4 4 NCCL_ALGO=Ring
echo "### NCCL collective log (rank 0 lines with ReduceScatter / algo), 32x1, 3 epochs  $(date +%H:%M:%S)"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,TUNING MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 8 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
    --data $P --grid 32 1 --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 3 --norm ln --fused --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
    | grep -i "NCCL INFO" | grep -i "reducescatter\|allgather\|algo\|proto\|PAT\|Ring\|Tree" | grep "\[0\]\|rank 0\|:0:" | sort | uniq -c | sort -rn | head -40 | cut -c1-220
echo "### done $(date +%H:%M:%S)"
