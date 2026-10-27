#!/bin/bash
# Graphormer-node with the probe's best recipe (lr 3e-4, dropout 0.1, max_degree 8) on Reddit and products.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 2400 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --fused --comm_bf16 --rs_bf16 --seed 0 --model graphormer --lr 3e-4 --dropout 0.1 --max_degree 8 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback" | cut -c1-420 | head -3
}
run "reddit h128 graphormer best"   --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 graphormer best" --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4
echo "### done $(date +%H:%M:%S)"
