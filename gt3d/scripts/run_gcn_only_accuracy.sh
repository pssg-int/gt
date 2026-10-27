#!/bin/bash
# Accuracy of the GCN-only model (--no_attn: no attention branch, plain layer) with the recipe of the attention-only /
# hybrid runs (scripts/run_attn_only_accuracy.sh), 4 GPUs 2x2, BatchNorm:
#   JID=$(salloc -A gen0034_g -q interactive -N 1 -C gpu --gpus-per-node 4 -t 01:00:00 --no-shell 2>&1 | grep -o '[0-9]*$')
#   bash scripts/run_gcn_only_accuracy.sh $JID
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 2400 srun --jobid=$JID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --no_attn "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-400 | head -3
}
for s in 0 1 2; do
  run "arxiv gcn-only seed $s" --epochs 600 --hidden 80 --heads 2 --layers 4 --seed $s
done
run "reddit h128 gcn-only"   --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 gcn-only" --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h256 gcn-only" --data $P --epochs 500 --eval_every 5 --hidden 256 --heads 8 --layers 4
echo "### done $(date +%H:%M:%S)"
