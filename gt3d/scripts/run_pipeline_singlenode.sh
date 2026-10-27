#!/bin/bash
# single-node check of the pipelined layer (pipe=1/2/4) and a 6-layer products accuracy probe
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
D=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
for ps in 1 2 4; do
  for cfg in "80 2 " "256 8 --checkpoint"; do
    set -- $cfg
    torchrun --nproc_per_node 4 --master_port 29650 gt3d/train/train_dist.py --data $D --grid 2 2 --backend triton --hidden $1 --heads $2 --layers 4 \
        --timing_epochs 12 --fused --comm_bf16 --rs_bf16 --pipeline_sub $ps $3 2>&1 | grep "^STEADY\|Error" | head -2
  done
done
echo "### products 6 layers hidden 128 dropout 0.3, 500 epochs"
torchrun --nproc_per_node 4 --master_port 29651 gt3d/train/train_dist.py --data $D --grid 2 2 --backend triton --hidden 128 --heads 4 --layers 6 \
    --dropout 0.3 --epochs 500 --eval_every 5 --checkpoint --fused --comm_bf16 2>&1 | grep "^FINAL\|Error" | sed -e 's/data=[^ ]* //' | cut -c1-220
