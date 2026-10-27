#!/bin/bash
# Accuracy with and without --tf32_matmul (TF32 tensor cores for the fp32 dense matmuls), one seed, the T3 recipe
# (4 GPUs 2x2, fused layer, bf16 gathers + RS, BatchNorm; RESULTS 2026-10-13 table):  JOBID=<alloc> bash scripts/run_tf32_accuracy.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 2400 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-420 | head -3
}
run "arxiv seed 0 ieee"        --epochs 600 --hidden 80 --heads 2 --layers 4 --seed 0
run "arxiv seed 0 tf32"        --epochs 600 --hidden 80 --heads 2 --layers 4 --seed 0 --tf32_matmul
run "reddit h128 ieee"         --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4 --seed 0
run "reddit h128 tf32"         --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4 --seed 0 --tf32_matmul
run "products h128 ieee"       --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4 --checkpoint --seed 0
run "products h128 tf32"       --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4 --checkpoint --seed 0 --tf32_matmul
echo "### done $(date +%H:%M:%S)"
