#!/bin/bash
#SBATCH -C "gpu&hbm40g"
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-sparse-256
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_sparse_comm_256_%j.log
# Sparsity-aware communication at 256 GPUs / 64 nodes (outline §4.7, Fig. 11): the first scale where the column share
# passes the plan's rule (products 64x4 share ~0.6, papers100M ~0.44; RESULTS 2026-10-23).  Three short jobs (PART):
#   PART=products : micro-benchmark 64x4 + 32x8; attention-only Exphormer h128 on 64x4 (dense / rule / forced), 32x8
#                   (dense / rule) and the 1D 256x1 (dense)                                        ~12 min
#   PART=papers64 : micro-benchmark papers 64x4; streamed h64 n_sub 16 on 64x4 and 32x8 (dense / rule)   ~15 min
#   PART=papers128: streamed h128 n_sub 32 (64x4) and n_sub 16 (32x8), dense / rule                 ~20 min
#   sbatch -q regular -N 64 -t 00:20:00 --export=ALL,PART=products  scripts/sbatch_sparse_comm_256.sh
#   sbatch -q regular -N 64 -t 00:25:00 --export=ALL,PART=papers64  scripts/sbatch_sparse_comm_256.sh
#   sbatch -q regular -N 64 -t 00:30:00 --export=ALL,PART=papers128 scripts/sbatch_sparse_comm_256.sh
# Memory model (grid_select.peak_memory_bytes, 256 GPUs): papers h64 64x4 n16 8.1 GB, 32x8 n16 8.5; h128 64x4 n32 11.7,
# 32x8 n16 15.5.  Model predictions: products 64x4 137 -> 114 ms (sparse), 32x8 96.5; papers h64 32x8 1.9 s, 64x4 2.9 s.
# With --sparse_comm the trainer skips its NCCL channel tuning on one-rank-per-node column groups (it slows a2a-v 1.5x).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
NODES=$SLURM_JOB_NUM_NODES; GP=$(( NODES * 4 ))
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; PAP=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
SRUN="srun -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores"
PART=${PART:-products}
echo "### PART=$PART nodes=$NODES gpus=$GP  $(date +%H:%M:%S)"
bench() {  # data grid_r grid_c timeout
  echo "### bench data=$(basename $1) gpus=$GP grid=$2x$3  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_sparse_comm.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout ${4:-600} $SRUN bash scripts/multinode_train.sh \
      --data $1 --grid $2 $3 --hidden 128 256 --heads 4 --shares 0.1 0.2 0.3 0.45 0.6 0.8 1.0 < /dev/null 2>&1 | grep "^\[bench\]\|^SPARSE\|^CURVE\|Traceback\|Error" | head -60
}
model() {  # data grid_r grid_c hidden heads timing_epochs timeout extra
  echo "### model data=$(basename $1) gpus=$GP grid=$2x$3 hidden=$4 heads=$5 $8  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout $7 $SRUN bash scripts/multinode_train.sh \
      --data $1 --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs $6 --norm ln --fused --comm_bf16 --rs_bf16 $8 < /dev/null 2>&1 \
      | grep "^STEADY\|^\[sparse_comm\]\|^\[nccl_tune\]\|^\[mem\] after backward (epoch 2)\|OutOfMemory\|Traceback\|Error" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0,1,300) }' | head -6
}
case $PART in
  products)
    bench $P 64 4 600
    bench $P 32 8 600
    model $P 64 4 128 4 10 600 "--no_gcn"
    model $P 64 4 128 4 10 600 "--no_gcn --sparse_comm"
    model $P 64 4 128 4 10 600 "--no_gcn --sparse_comm --sparse_threshold 1.0 --sparse_threshold_row 0"
    model $P 32 8 128 4 10 600 "--no_gcn"
    model $P 32 8 128 4 10 600 "--no_gcn --sparse_comm"
    model $P 256 1 128 4 10 600 "--no_gcn"
    ;;
  papers64)
    bench $PAP 64 4 900
    model $PAP 64 4 64 4 5 900 "--stream_sub 16 --checkpoint"
    model $PAP 64 4 64 4 5 900 "--stream_sub 16 --checkpoint --sparse_comm"
    model $PAP 32 8 64 4 5 900 "--stream_sub 16 --checkpoint"
    model $PAP 32 8 64 4 5 900 "--stream_sub 16 --checkpoint --sparse_comm"
    ;;
  papers128)
    # 64x4: the chunk (433,840 rows at 256 ranks) is divisible by 16 but not by 32 (job 60312017 hit the assertion)
    model $PAP 64 4 128 4 4 1200 "--stream_sub 16 --checkpoint"
    model $PAP 64 4 128 4 4 1200 "--stream_sub 16 --checkpoint --sparse_comm"
    model $PAP 32 8 128 4 4 1200 "--stream_sub 16 --checkpoint"
    model $PAP 32 8 128 4 4 1200 "--stream_sub 16 --checkpoint --sparse_comm"
    ;;
  papers128b)   # the 64x4 pair only (resubmission after 60312017)
    model $PAP 64 4 128 4 4 1200 "--stream_sub 16 --checkpoint"
    model $PAP 64 4 128 4 4 1200 "--stream_sub 16 --checkpoint --sparse_comm"
    ;;
  *) echo "unknown PART $PART"; exit 1;;
esac
echo "### done $(date +%H:%M:%S)"
