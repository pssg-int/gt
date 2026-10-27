#!/bin/bash
# Sparse communication on the real products tiles, one node / 4 GPUs: smoke test + timing (no gain expected at P_d <= 4:
# the tiles reference 97-100 % of the blocks; this checks the real-graph path, the plan build time and the forced-sparse penalty).
#   srun --jobid=<alloc> -N1 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh ... is wrapped below
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
JOB=${JOB:?set JOB=<allocation id>}
run() {  # grid_r grid_c extra
  echo "### products 4 GPUs grid=$1x$2 $3  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29800 + RANDOM % 100 )) timeout 600 srun --jobid=$JOB -N1 --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh \
      --data $P --grid $1 $2 --backend triton --hidden 128 --heads 4 --layers 4 --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 $3 < /dev/null 2>&1 \
      | grep "^STEADY\|^\[sparse_comm\]\|^data:\|graph structures\|OutOfMemory\|Traceback\|Error" | awk '{ if ($0 ~ /^STEADY/) print; else print substr($0,1,260) }' | head -6
}
run 2 2 "--no_gcn"
run 2 2 "--no_gcn --sparse_comm --sparse_threshold 1.0"
run 4 1 "--no_gcn"
run 4 1 "--no_gcn --sparse_comm --sparse_threshold 1.0"
run 4 1 "--no_gcn --sparse_comm"
run 2 2 "--stream_sub 4"
run 2 2 "--stream_sub 4 --sparse_comm --sparse_threshold 1.0"
echo "### done $(date +%H:%M:%S)"
