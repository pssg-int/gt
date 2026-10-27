#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 16
#SBATCH --gpus-per-node 4
#SBATCH -t 00:25:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-e11-64
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_variants_64_%j.log
# E11 generality: Exphormer / UniMP / Graphormer-node on the same system at 64 GPUs (16x4, fused layer, bf16 gathers + RS,
# no checkpoint, 10 timing epochs; LN for Exphormer, the variants normalise per node by construction).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # label data hidden heads extra...
  label=$1; data=$2; hid=$3; heads=$4; shift 4
  echo "### $label  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 600 srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $data --grid 16 4 --backend triton --hidden $hid --heads $heads --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 --timers "$@" < /dev/null 2>&1 \
      | grep "^data:\|^\[mem\] after backward (epoch 2)\|^STEADY\|Max Time\|Error\|error:\|Traceback\|OutOfMemory" | grep -v "^|--\[95m 'stream\|^|--\[95m 'attn" | cut -c1-200 | head -40
}
run "products h128 exphormer"  $P 128 4 --model exphormer
run "products h128 unimp"      $P 128 4 --model unimp
run "products h128 graphormer" $P 128 4 --model graphormer
run "reddit h128 exphormer"    $R 128 4 --model exphormer
run "reddit h128 unimp"        $R 128 4 --model unimp
run "reddit h128 graphormer"   $R 128 4 --model graphormer
run "products h256 exphormer"  $P 256 8 --model exphormer
run "products h256 unimp"      $P 256 8 --model unimp
run "products h256 graphormer" $P 256 8 --model graphormer
echo "### done $(date +%H:%M:%S)"
