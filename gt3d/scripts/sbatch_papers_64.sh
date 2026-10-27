#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:55:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-papers64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_papers_64_%j.log
# ogbn-papers100M full-graph Exphormer on 64 GPUs, timing only (5 epochs: 2 autotuning + 3 steady), after the
# row-tiled kernels (sparse column sub-blocks), int32 graph structures and expandable-segments allocator.
# Reference (2026-09-26, per-row kernels): hidden 64 16x4 n_sub 16 13955 ms / 34.2 GB, 32x2 13551 / 30.3, 64x1 n_sub 32
# 15482 / 30.3; hidden 128 OOM everywhere (32x2 n_sub 64: 29.8 GB allocated + 7.3 GB reserved).
# Results of job 59781950 (2026-10-15): RESULTS.md 2026-10-15 section; --stream_overlap is the default since then
# (-21..-28 % here), so the "--stream_overlap" runs below are now the plain configuration (use --no_stream_overlap for the old one).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
run() {  # grid_r grid_c hidden heads stream_sub extra
  echo "### nodes=$SLURM_JOB_NUM_NODES gpus=$(( SLURM_JOB_NUM_NODES * 4 )) grid=$1x$2 hidden=$3 heads=$4 stream=$5 $6  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $D --grid $1 $2 --backend triton --hidden $3 --heads $4 --layers 4 --timing_epochs 5 --norm ln --checkpoint --fused --comm_bf16 --rs_bf16 --stream_sub $5 --timers $6 < /dev/null 2>&1 \
      | grep "^data:\|^\[mem\]\|^STEADY\|^\[timers\]\|Max Time\|Error\|error:\|Traceback\|OutOfMemory" | cut -c1-200 | head -80
}
run 16 4 64 4 16 ""
run 16 4 64 4 32 ""
run 32 2 64 4 16 ""
run 64 1 64 4 32 ""
run 32 2 128 4 64 ""
run 64 1 128 4 64 ""
run 16 4 128 4 64 ""
# communication / compute overlap in the streamed layer (prefetch of the next column sub-block, async dx reduce-scatter)
run 16 4 64 4 16 "--stream_overlap"
run 16 4 64 4 32 "--stream_overlap"
run 32 2 128 4 64 "--stream_overlap"
# tensor-core precision for the tiled kernels (only if its correctness test has passed by the time the job runs)
if [ -f logs/tiled_prec_tf32x2_ok ]; then
  export GT3D_TILED_PREC=tf32x2
  run 16 4 64 4 16 "--stream_overlap"
  run 32 2 128 4 64 "--stream_overlap"
  unset GT3D_TILED_PREC
fi
echo "### done $(date +%H:%M:%S)"
