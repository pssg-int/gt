#!/bin/bash
# Graphormer-node on arxiv: learning-rate / dropout probes (the T3 recipe gave 49 % test; pre-LN transformers usually
# train with a smaller lr).  JOBID=<alloc> bash scripts/run_graphormer_probe.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
run() {  # label  args...
  label=$1; shift
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 1200 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --fused --comm_bf16 --rs_bf16 --seed 0 --model graphormer --hidden 80 --heads 2 --layers 4 "$@" < /dev/null 2>&1 \
      | grep "^FINAL\|Error\|OutOfMemory\|Traceback" | sed 's/.*best_val=\([0-9.]*\) test@best_val=\([0-9.]*\) epoch=\([0-9]*\).*/  val \1 test \2 epoch \3/' | head -3
}
run "lr 1e-3 dropout 0.3 (T3, reference 49.2)" --epochs 600
run "lr 3e-4 dropout 0.3"              --epochs 600 --lr 3e-4
run "lr 3e-4 dropout 0.1"              --epochs 600 --lr 3e-4 --dropout 0.1
run "lr 1e-4 dropout 0.1 1000 ep"      --epochs 1000 --lr 1e-4 --dropout 0.1
run "lr 3e-4 dropout 0.1 max_degree 8" --epochs 600 --lr 3e-4 --dropout 0.1 --max_degree 8
run "lr 3e-4 dropout 0.1 no weight decay" --epochs 600 --lr 3e-4 --dropout 0.1 --weight_decay 0
echo "### done $(date +%H:%M:%S)"
