#!/bin/bash
# Full-graph products accuracy runs on 4 GPUs (2x2), two wider models, 500 epochs each.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for cfg in "128 4 " "256 8 --checkpoint"; do
  set -- $cfg
  echo "### products accuracy hidden=$1 heads=$2 $3  $(date +%H:%M:%S)"
  torchrun --nproc_per_node 4 --master_port 29550 gt3d/train/train_dist.py --data $D --grid 2 2 --backend triton \
      --hidden $1 --heads $2 --layers 4 --epochs 500 --eval_every 5 $3 2>&1 | grep "^epoch  *[0-9]*00 \|^FINAL\|Error" | cut -c1-220
done
