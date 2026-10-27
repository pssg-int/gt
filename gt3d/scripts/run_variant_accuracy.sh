#!/bin/bash
# Accuracy of the UniMP / Graphormer-node variants with the T3 recipe of the Exphormer table (4 GPUs 2x2, fused layer,
# bf16 gathers + RS, same depth / width / optimiser, seed 0):  JOBID=<alloc> bash scripts/run_variant_accuracy.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 2400 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --fused --comm_bf16 --rs_bf16 --seed 0 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|^data:\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-420 | head -4
}
run "arxiv unimp"                 --model unimp --epochs 600 --hidden 80 --heads 2 --layers 4
run "arxiv unimp label_input"     --model unimp --label_input --epochs 600 --hidden 80 --heads 2 --layers 4
run "arxiv graphormer"            --model graphormer --epochs 600 --hidden 80 --heads 2 --layers 4
run "reddit h128 unimp"           --model unimp --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "reddit h128 graphormer"      --model graphormer --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 unimp label_input" --model unimp --label_input --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 graphormer"    --model graphormer --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 unimp"         --model unimp --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4
echo "### done $(date +%H:%M:%S)"
