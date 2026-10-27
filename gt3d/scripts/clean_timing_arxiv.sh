#!/bin/bash
# Sequential, uncontended 100-epoch timing runs on one GPU: ours (triton / torch backends) and the official code.
set -x
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
python -u gt3d/train/train_single_gpu.py --backend triton --epochs 100 --eval_every 1
python -u gt3d/train/train_single_gpu.py --backend torch --epochs 100 --eval_every 1
cd /pscratch/sd/c/cunyang/gnn/baseline/Exphormer
python -u main.py --cfg configs/Exphormer/arxiv.yaml --repeat 1 wandb.use False seed 0 out_dir runs/results_timing optim.max_epoch 100 2>&1 | grep -E "Epoch 99|Avg time per epoch|Num parameters"
