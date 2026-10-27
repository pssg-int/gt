#!/bin/bash
# Accuracy of the attention-only Exphormer (--no_gcn: no parallel GCN branch) vs the full GCN+attention model, same recipe,
# 4 GPUs (2x2 grid, fused layer, bf16 communication, BatchNorm), inside a 1-node interactive allocation:
#   JID=$(salloc -A gen0034_g -q interactive -N 1 -C gpu --gpus-per-node 4 -t 02:00:00 --no-shell 2>&1 | grep -o '[0-9]*$')
#   bash scripts/run_attn_only_accuracy.sh $JID
# arxiv: official recipe (600 epochs, hidden 80 / 2 heads / 4 layers, dropout 0.3), 3 seeds per arm;
# Reddit: hidden 128 / 4 heads, 400 epochs; products: hidden 128 / 4 heads, 500 epochs (+ hidden 256 attention-only).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 2400 srun --jobid=$JID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-400 | head -3
}
for s in 0 1 2; do
  run "arxiv attn-only seed $s" --epochs 600 --hidden 80 --heads 2 --layers 4 --seed $s --no_gcn
  run "arxiv gcn+attn seed $s"  --epochs 600 --hidden 80 --heads 2 --layers 4 --seed $s
done
run "reddit h128 attn-only" --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4 --no_gcn
run "reddit h128 gcn+attn"  --data $R --epochs 400 --eval_every 5 --hidden 128 --heads 4 --layers 4
run "products h128 attn-only" --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4 --checkpoint --no_gcn
run "products h128 gcn+attn"  --data $P --epochs 500 --eval_every 5 --hidden 128 --heads 4 --layers 4 --checkpoint
run "products h256 attn-only" --data $P --epochs 500 --eval_every 5 --hidden 256 --heads 8 --layers 4 --checkpoint --no_gcn
echo "### done $(date +%H:%M:%S)"
