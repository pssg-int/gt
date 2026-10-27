#!/bin/bash
# 4-GPU (one node) half of the gather_proj comparison (the 8 / 16 GPU half is scripts/sbatch_gather_proj.sh):
#   srun --jobid=<id> -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh bash scripts/run_gather_proj_4gpu.sh   (NO: needs 4 tasks)
# Run from a login shell with an allocation:  JOBID=<id> bash scripts/run_gather_proj_4gpu.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data hidden heads extra
  echo "### data=$(basename $1) nodes=1 gpus=4 grid=2x2 hidden=$2 heads=$3 $4  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 400 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid 2 2 --backend triton --hidden $2 --heads $3 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $4 < /dev/null 2>&1 \
      | grep "^STEADY\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
bench() {  # data hidden heads strategy
  echo "### ATTN data=$(basename $1) nodes=1 gpus=4 grid=2x2 hidden=$2 heads=$3 strategy=$4  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 300 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $1 --grid 2 2 --strategy $4 --hidden $2 --heads $3 --iters 10 --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
      | grep "^ATTN\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
run $P 80 2 "";   run $P 80 2 "--gather_proj"
run $P 128 4 "";  run $P 128 4 "--gather_proj"
run $P 256 8 "--checkpoint";  run $P 256 8 "--checkpoint --gather_proj"
run $R 128 4 "";  run $R 128 4 "--gather_proj"
run $R 256 8 "";  run $R 256 8 "--gather_proj"
for cfg in "80 2" "256 8"; do set -- $cfg; bench $P $1 $2 gather; bench $P $1 $2 gather_proj; done
bench $R 256 8 gather; bench $R 256 8 gather_proj
echo "### done $(date +%H:%M:%S)"
