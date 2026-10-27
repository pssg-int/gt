#!/bin/bash
# Quiet-node timings (run with the GPUs otherwise idle):  JOBID=<alloc> bash scripts/run_quiet_timings.sh
#  1. tiled-kernel micro-benchmark per precision mode (ieee / tf32x2 / tf32)
#  2. streamed layer with / without --stream_overlap, products 4 GPUs (no timers)
#  3. the products h128 gather_proj pair again (its first run shared the node with kernel compilations)
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for prec in ieee tf32x2 tf32; do
  echo "### bench_row_overhead GT3D_TILED_PREC=$prec $(date +%H:%M:%S)"
  GT3D_TILED_PREC=$prec timeout 1500 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh \
      python bench/bench_row_overhead.py --edges 3500000 7000000 14000000 28000000 --block_r 16 2>&1 | grep -v '^\[run_on_node\]'
done
run() {  # hidden heads extra
  echo "### products 2x2 hidden=$1 heads=$2 $3  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $P --grid 2 2 --backend triton --hidden $1 --heads $2 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 $3 < /dev/null 2>&1 \
      | grep "^STEADY\|OutOfMemory\|Traceback\|Error" | cut -c1-200 | head -3
}
for ov in "" "--stream_overlap"; do run 256 8 "--checkpoint --stream_sub 16 $ov"; run 256 8 "--checkpoint --stream_sub 4 $ov"; done
run 256 8 "--checkpoint --stream_sub 16 --stream_overlap"
GT3D_TILED_PREC=tf32x2 run 256 8 "--checkpoint --stream_sub 16 --stream_overlap"
run 128 4 ""; run 128 4 "--gather_proj"; run 128 4 ""; run 128 4 "--gather_proj"
echo "### done $(date +%H:%M:%S)"
