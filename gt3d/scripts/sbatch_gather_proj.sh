#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J gt3d-gproj
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_gather_proj_%j.log
# "Compute for bytes": fused layer gathering x (width d) and projecting the gathered blocks redundantly on every rank
# vs --gather_proj (owner projects once; Q gathered over the row group, [K|V|GCN msg] (width 3d) over the column
# group).  Full model (LN, bf16 gathers + RS, 10 timing epochs) and the attention-only operator (bench), 8 / 16 GPUs.
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
run() {  # data nodes grid_r grid_c hidden heads extra
  echo "### data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 hidden=$5 heads=$6 $7  $(date +%H:%M:%S)"
  MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 300 srun -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh \
      --data $1 --grid $3 $4 --backend triton --hidden $5 --heads $6 --layers 4 --timing_epochs 10 --norm ln --fused --comm_bf16 --rs_bf16 $7 < /dev/null 2>&1 \
      | grep "^STEADY\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
bench() {  # data nodes grid_r grid_c hidden heads strategy
  echo "### ATTN data=$(basename $1) nodes=$2 gpus=$(( $2 * 4 )) grid=$3x$4 hidden=$5 heads=$6 strategy=$7  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 300 srun -N $2 --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores \
      bash scripts/multinode_train.sh --data $1 --grid $3 $4 --strategy $7 --hidden $5 --heads $6 --iters 10 --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
      | grep "^ATTN\|OutOfMemory\|Traceback\|Error" | cut -c1-300 | head -3
}
for cfg in "80 2" "128 4" "256 8"; do set -- $cfg
  run $P 2 2 4 $1 $2 "";  run $P 2 2 4 $1 $2 "--gather_proj"
  run $P 4 4 4 $1 $2 "";  run $P 4 4 4 $1 $2 "--gather_proj"
done
for cfg in "128 4" "256 8"; do set -- $cfg
  run $R 2 2 4 $1 $2 "";  run $R 2 2 4 $1 $2 "--gather_proj"
  run $R 4 4 4 $1 $2 "";  run $R 4 4 4 $1 $2 "--gather_proj"
done
for cfg in "80 2" "256 8"; do set -- $cfg
  bench $P 2 2 4 $1 $2 gather;  bench $P 2 2 4 $1 $2 gather_proj
  bench $P 4 4 4 $1 $2 gather;  bench $P 4 4 4 $1 $2 gather_proj
done
bench $R 4 4 4 256 8 gather;  bench $R 4 4 4 256 8 gather_proj
echo "### done $(date +%H:%M:%S)"
