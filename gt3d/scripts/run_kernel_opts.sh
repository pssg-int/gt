#!/bin/bash
# after the kernel unit tests: option microbenchmark, model tests (fp64 / fp32 / bf16 operands), arxiv accuracy, products timings
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
while [ "$(grep -c 'run_on_node\] exit=' logs/kernel_tests_$JOBID.log)" -lt 2 ]; do sleep 15; done
echo "### kernel option microbenchmark $(date +%H:%M:%S)"
timeout 1800 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh python bench/bench_kernel_opts.py 2>&1 | grep -v 'Warning\|warn\|run_on_node'
echo "### model tests $(date +%H:%M:%S)"
for args in "--grid 2 2 --backend torch --fused" "--grid 2 2 --backend triton --fused" "--grid 2 2 --backend triton --fused --kernel_bf16" \
            "--grid 4 1 --backend triton --fused --kernel_bf16" "--grid 2 2 --backend triton --fused --stream_sub 4 --checkpoint" \
            "--grid 2 2 --backend triton --fused --stream_sub 4 --checkpoint --kernel_bf16" "--grid 2 2 --backend triton --fused --stream_sub 8 --kernel_bf16"; do
  echo "--- test_dist_model $args"
  MASTER_PORT=$(( 29800 + RANDOM % 100 )) timeout 900 srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node 4 bash scripts/run_on_node.sh \
      torchrun --nproc_per_node 4 --master_port $(( 29900 + RANDOM % 90 )) tests/test_dist_model.py $args 2>&1 | grep "dist-model rank 0\|ALL PASSED\|FAILED\|Traceback\|Error" | grep -v warn | cut -c1-230
done
run() {  # label hidden heads extra...
  local label=$1 hid=$2 heads=$3; shift 3
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --grid 2 2 --backend triton --hidden $hid --heads $heads --layers 4 \
      --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep -i "STEADY\|attn fwd kernel\|attn bwd kernel\|'forward\|'backward\|after backward (epoch 2)\|OutOfMemory\|Traceback\|error" | grep -v 'warn\|^|--\[95m .attn' | cut -c1-200
}
echo "### products 2x2 timings (TF32; heads/program 4 vs 8; bf16 operands) $(date +%H:%M:%S)"
GT3D_HEADS_PER_PROGRAM=8 run "h256 hpp8 fp32" 256 8 --checkpoint
GT3D_HEADS_PER_PROGRAM=4 run "h256 hpp4 fp32" 256 8 --checkpoint
GT3D_HEADS_PER_PROGRAM=4 run "h256 hpp4 bf16" 256 8 --checkpoint --kernel_bf16
GT3D_HEADS_PER_PROGRAM=4 run "h256 hpp4 bf16 no-ckpt" 256 8 --kernel_bf16
run "h128 fp32" 128 4
run "h128 bf16" 128 4 --kernel_bf16
run "h80 fp32" 80 2
run "h80 bf16" 80 2 --kernel_bf16
echo "### arxiv accuracy (600 epochs, 4 GPUs 2x2): baseline vs --kernel_bf16 $(date +%H:%M:%S)"
for kb in "" "--kernel_bf16"; do
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --grid 2 2 --backend triton --hidden 80 --heads 2 --layers 4 --epochs 600 --norm ln --fused --comm_bf16 --rs_bf16 --eval_every 5 $kb < /dev/null 2>&1 \
      | grep "^FINAL\|Traceback" | cut -c1-260
done
echo "### done $(date +%H:%M:%S)"
