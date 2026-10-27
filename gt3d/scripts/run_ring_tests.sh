#!/bin/bash
# Ring attention: correctness tests (fp64 vs the single-process reference) and quick one-node timings.
#   srun -A gen0034_g -q interactive -N 1 -C gpu --gpus 4 -t 00:40:00 bash scripts/run_on_node.sh bash scripts/run_ring_tests.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
t() {  # nproc  args...
  np=$1; shift
  res=$(timeout 600 torchrun --nproc_per_node $np --master_port $(( 29500 + RANDOM % 300 )) tests/test_dist_model.py "$@" 2>&1)
  echo "### test np=$np $* -> $(echo "$res" | grep -o "\[dist-model\] ALL PASSED\|\[dist-model\] FAILED" | head -1)"
  echo "$res" | grep "dist-model rank\|Error\|error\|Traceback" | grep -v ChildFailed | head -8
}
t 4 --grid 4 1 --ring --backend torch
t 4 --grid 4 1 --ring --backend triton
t 4 --grid 4 1 --ring --backend torch --checkpoint
t 2 --grid 2 1 --ring --backend torch
t 1 --grid 1 1 --ring --backend torch
echo "### arxiv smoke (4 GPUs, ring vs 1D GP-AG vs 2x2), 5 timing epochs"
for cfg in "--grid 4 1 --ring --comm_bf16" "--grid 4 1 --fused --comm_bf16 --rs_bf16" "--grid 2 2 --fused --comm_bf16 --rs_bf16"; do
  timeout 600 torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) gt3d/train/train_dist.py $cfg --backend triton --timing_epochs 5 --norm ln 2>&1 | grep "^STEADY\|Error\|Traceback" | cut -c1-250 | head -3
done
run() {  # data grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) grid=$2x$3 hidden=$4 heads=$5 $6  $(date +%H:%M:%S)"
  timeout 900 torchrun --nproc_per_node 4 --master_port $(( 29500 + RANDOM % 300 )) gt3d/train/train_dist.py \
      --data $1 --grid $2 $3 --backend triton --hidden $4 --heads $5 --layers 4 --timing_epochs 10 --norm ln --timers $6 2>&1 \
      | grep "^STEADY\|ring fwd\|ring bwd\|^\[mem\] after backward (epoch 2)\|Error\|OutOfMemory\|Traceback" | cut -c1-220 | head -12
}
for cfg in "80 2" "256 8 --checkpoint"; do set -- $cfg; hd=$1; hh=$2; shift 2; ex="$*"
  run $P 4 1 $hd $hh "--ring --comm_bf16 $ex"
  run $P 4 1 $hd $hh "--fused --comm_bf16 --rs_bf16 $ex"
  run $P 2 2 $hd $hh "--fused --comm_bf16 --rs_bf16 $ex"
done
run $R 4 1 128 4 "--ring --comm_bf16"
run $R 4 1 128 4 "--fused --comm_bf16 --rs_bf16"
run $R 2 2 128 4 "--fused --comm_bf16 --rs_bf16"
echo "### done $(date +%H:%M:%S)"
