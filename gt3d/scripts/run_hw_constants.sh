#!/bin/bash
# Hardware constants + per-collective backward timers + saved-tensor census, on an interactive 4-node allocation:
#   bash scripts/run_hw_constants.sh <jobid>   (log: logs/hw_constants_<jobid>.log)
JOBID=$1
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b
RD=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # nodes data grid-rows grid-cols hidden heads extra...
  local nodes=$1 data=$2 pr=$3 pc=$4 hid=$5 heads=$6; shift 6
  echo "### nodes=$nodes data=$(basename $data) grid=${pr}x${pc} hidden=$hid $*  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 srun --jobid=$JOBID -N $nodes --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $data --grid $pr $pc --backend triton --hidden $hid --heads $heads --layers 4 \
      --timing_epochs 6 --norm ln --fused --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep -v '^\[multinode\]\|Warning\|warn' | grep "STEADY\|Max Time\|\[census\]\|\[mem\]\|OutOfMemory\|Traceback\|Error" | cut -c1-220
}
# backward collective timers (new in grid2d) at 16 / 8 / 4 GPUs
run 4 $P 4 4 80 2 --timers
run 4 $P 4 4 128 4 --timers
run 4 $P 4 4 256 8 --timers
run 4 $RD 4 4 128 4 --timers
run 4 $RD 4 4 256 8 --timers
run 2 $P 2 4 80 2 --timers
run 2 $P 2 4 128 4 --timers
run 1 $P 2 2 80 2 --timers
run 1 $P 2 2 128 4 --timers
# saved-tensor census (8 GPUs 2x4: O, R, C all distinct)
run 2 $P 2 4 80 2 --saved_census --timing_epochs 3
run 2 $P 2 4 128 4 --saved_census --timing_epochs 3
run 2 $RD 2 4 128 4 --saved_census --timing_epochs 3
echo "### done $(date +%H:%M:%S)"
