#!/bin/bash
# Accuracy comparison, 3 seeds: ours (triton backend) on GPUs 1-3 in parallel, official sequentially on GPU 0.
G=/pscratch/sd/c/cunyang/gnn/gt/gt3d
cd $G
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s+1)) python -u gt3d/train/train_single_gpu.py --backend triton --epochs 600 --seed $s > logs/acc_ours_triton_s$s.log 2>&1 &
done
cd /pscratch/sd/c/cunyang/gnn/baseline/Exphormer
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -u main.py --cfg configs/Exphormer/arxiv.yaml --repeat 1 wandb.use False seed $s out_dir runs/results_acc > runs/acc_official_s$s.log 2>&1
done
wait
echo "=== ours ==="; grep -h "^FINAL" $G/logs/acc_ours_triton_s*.log
echo "=== official ==="; for s in 0 1 2; do grep "^> Epoch 599" runs/acc_official_s$s.log | sed "s/^/seed $s: /"; done
