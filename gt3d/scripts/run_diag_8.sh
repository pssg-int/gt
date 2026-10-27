#!/bin/bash
# backward-collective diagnosis at 8 GPUs (2 nodes): CPU enqueue times + profiler trace;  bash scripts/run_diag_8.sh <jobid> <nodes>
JOBID=$1; NN=${2:-2}
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # grid-rows grid-cols hidden heads extra...
  local pr=$1 pc=$2 hid=$3 heads=$4; shift 4
  echo "### nodes=$NN grid=${pr}x${pc} hidden=$hid $*  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $NN --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $P --grid $pr $pc --backend triton --hidden $hid --heads $heads --layers 4 \
      --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep -i "STEADY\|Max Time\|\[cpu\]\|\[prof\]\|OutOfMemory\|Traceback\|error" | grep -v 'warn' | cut -c1-220
}
run 2 4 128 4 --timers
GT3D_PROF_TRACE=/pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/trace8_products_h128 run 2 4 128 4 --profile --timing_epochs 3
echo "### done $(date +%H:%M:%S)"
