#!/bin/bash
# Follow-ups of the head-parallel study: compute-only (--no_comm) and device-time profiles (--profile) for selected grids.
#   bash scripts/run_attn_head_extra.sh <jobid> <nodes>      (nodes = 2 -> 8-GPU set, 4 -> 16-GPU set)
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
JID=$1; NODES=$2
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
run() {  # strategy hidden heads extra grid...
  strat=$1; hid=$2; hds=$3; extra=$4; shift 4
  echo "### data=products_p16b nodes=$NODES gpus=$(( NODES * 4 )) strategy=$strat grid=$* hidden=$hid heads=$hds extra=$extra  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JID -N $NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $P --grid $* --strategy $strat --hidden $hid --heads $hds --iters 10 --comm_bf16 --rs_bf16 $extra < /dev/null 2>&1 \
      | grep "^ATTN\|^PROF\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-400 | head -4
}
if [ "$NODES" = "2" ]; then
  run gather 256 8 --no_comm 2 4
  run head 256 8 --no_comm 1 1 8
  run head 256 8 --no_comm 2 1 4
  run head 256 8 --no_comm 2 2 2
  run head 256 8 --profile 1 1 8
  run head 256 8 --profile 2 1 4
  run gather 256 8 --profile 2 4
else
  run gather 256 8 --no_comm 4 4
  run head 256 8 --no_comm 4 4 1
  run head 256 8 --no_comm 4 2 2
  run head 256 8 --no_comm 2 4 2
  run head 256 8 --no_comm 4 1 4
  run head 256 8 --no_comm 2 1 8
  run head 256 8 --profile 4 1 4
  run head 256 8 --profile 2 4 2
  run head 256 8 --profile 2 1 8
  run head 256 8 --profile 4 2 2
  run gather 256 8 --profile 4 4
  run head 80 2 --no_comm 2 4 2
  run head 80 2 --no_comm 4 2 2
fi
echo "### done $(date +%H:%M:%S)"
