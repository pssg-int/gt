#!/bin/bash
# ogbn-products full graph on 4 GPUs (2x2 grid): timing runs, with and without activation checkpointing.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for cfg in "80 2 4 " "80 2 4 --checkpoint" "256 8 4 --checkpoint"; do
  set -- $cfg
  echo "### products 2x2 hidden=$1 heads=$2 layers=$3 $4"
  torchrun --nproc_per_node 4 --master_port 29540 gt3d/train/train_dist.py --data $D --grid 2 2 --backend triton \
      --hidden $1 --heads $2 --layers $3 --timing_epochs 10 $4 2>&1 | grep "^\[mem\]\|^FINAL\|Error\|error\|Traceback" | cut -c1-260 | head -8
done
