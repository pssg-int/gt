#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-hw-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_hw_64_%j.log
# White-box model inputs at 64 GPUs: NCCL constants with a 16-node group (bench/hw_constants.py), the fused layer's
# per-collective backward timers, CPU-side backward enqueue times, and a profiler trace of products h128.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
echo "### hw_constants 64 ranks $(date +%H:%M:%S)"
MASTER_PORT=29650 GT3D_SCRIPT=bench/hw_constants.py timeout 900 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
    bash scripts/multinode_train.sh < /dev/null 2>&1 | grep -v 'Warning\|warn\|^\[rank\|sparse_csr_tensor\|multinode' | cut -c1-200
cp bench/hw_constants.json bench/hw_constants_64.json
run() {  # label data hidden heads extra...
  label=$1; data=$2; hid=$3; heads=$4; shift 4
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N 16 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --grid 16 4 --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 8 --norm ln --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep -i "^data:\|^\[mem\] after backward (epoch 2)\|^STEADY\|Max Time\|\[cpu\]\|\[prof\]\|\[census\]\|error\|Traceback\|OutOfMemory" | cut -c1-200 | head -60
}
run "products h128 timers" $P 128 4 --timers
run "products h256 timers" $P 256 8 --timers
run "reddit h128 timers"   $R 128 4 --timers
run "products h80 timers"  $P 80 2 --timers
run "products h128 8x8 timers" $P 128 4 --timers --grid 8 8
run "products h128 census" $P 128 4 --saved_census --timing_epochs 3
GT3D_PROF_TRACE=/pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/trace64_products_h128 run "products h128 profile" $P 128 4 --profile --timing_epochs 3
echo "### done $(date +%H:%M:%S)"
