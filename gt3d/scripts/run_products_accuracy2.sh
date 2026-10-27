#!/bin/bash
# Products accuracy, second round: wider model, stronger regularisation, longer schedule (4 GPUs, 2x2, fused + bf16 comm).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for cfg in "256 8 0.5 800" "128 4 0.5 800"; do
  set -- $cfg
  echo "### products accuracy hidden=$1 heads=$2 dropout=$3 epochs=$4  $(date +%H:%M:%S)"
  torchrun --nproc_per_node 4 --master_port 29620 gt3d/train/train_dist.py --data $D --grid 2 2 --backend triton \
      --hidden $1 --heads $2 --layers 4 --dropout $3 --epochs $4 --eval_every 5 --checkpoint --fused --comm_bf16 2>&1 \
      | grep "^epoch  *[0-9]*00 \|^FINAL\|Error" | sed -e 's/data=[^ ]* //' | cut -c1-220
done
