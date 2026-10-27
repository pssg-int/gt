#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:20:00
#SBATCH -A m2404
#SBATCH -J gt3d-integration-check
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_integration_check_%j.log
# After the Plexus/AxoNN integration: 16-GPU steady-state times must match the LN final table (Reddit h128 90.8 ms,
# products h80 368 ms, h256 ckpt 1555 ms at 4x4) within noise; one run with --timers prints the AxoNN region tree;
# one run reads the Plexus partition of Reddit directly.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
RP=/pscratch/sd/c/cunyang/gnn/plexus/dataset/reddit/processed_part4
run() {  # data hidden heads extra
  echo "### data=$(basename $1) 16 GPUs 4x4 hidden=$2 $4  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 500 srun -N 4 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid 4 4 --backend triton --hidden $2 --heads $3 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $4 2>&1 \
      | grep -v "Warning\|warn" | grep "^STEADY\|Max Time\|timers\|Error\|OutOfMemory\|Traceback" | cut -c1-200 | head -40
}
echo "### loader tests (CPU/gloo, node 1): Plexus partition 4x4 + 2x2, gt3d partition 1x32 / 8x4 (P > p_part)"
srun -N 1 -n 1 -c 64 bash scripts/run_on_node.sh bash -c 'for g in "4 4" "2 2"; do set -- $g; torchrun --nproc_per_node $(( $1 * $2 )) --master_port 29811 tests/test_plexus_loader.py --grid $1 $2 2>&1 | grep "src_first\|PASSED\|FAIL\|Error"; done; for g in "1 32" "8 4"; do torchrun --nproc_per_node 32 --master_port 29812 tests/test_loader_ranges.py --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --grid $g 2>&1 | grep "nodes\|PASSED\|FAIL\|Error"; done' 2>&1 | grep -v "Warning\|warn" | cut -c1-200
run $R 128 4 ""
run $R 128 4 "--timers"
run $RP 128 4 ""
run $P 80 2 ""
run $P 80 2 "--timers"
run $P 256 8 "--checkpoint"
echo "### done $(date +%H:%M:%S)"
