#!/bin/bash
# DAC'26 baselines (Lin, Madduri, Kandemir, "Scalable and Adaptive Parallel Training of Graph Transformer on Large
# Graphs", arXiv 2604.16715; not open source) re-implemented on our kernel, timed next to our layouts in one allocation:
#   GP-AG    (paper Alg. 1): head path, grid (P,1,1): owners project Q/K/V, all-gather K and V (2 AG + 2 RS)
#   GP-AG-x  (our variant):  fused layer, grid (P,1): all-gather x, every rank re-projects the gathered K/V
#   GP-A2A   (paper Alg. 2): head path, grid (1,1,P), P <= heads: a2a of Q, K, V and of the output (8 a2a per layer)
#   ours:    fused 2D and head-grid candidates of our selector; the full model also runs --head_auto
# Models: hidden 128 / 8 heads (the paper's setting) and hidden 256 / 8 heads; bf16 communication everywhere.
#
#   bash scripts/run_dac26_strategies.sh <jobid | -> <attn | model | all> <nodes>...
# jobid "-" = inside an sbatch script.  REL=<k> places every step on the allocation's nodes k, k+1, ... (concurrent
# steps on disjoint nodes of one interactive allocation).
JOBID=$1; PART=$2; shift 2
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
P=/pscratch/sd/c/cunyang/gnn/gt/data/products_p16b; R=/pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16
SRUN="srun"; [ "$JOBID" != "-" ] && SRUN="srun --jobid=$JOBID"
[ -n "$REL" ] && SRUN="$SRUN --relative=$REL"

attn() {  # label nodes data hidden heads strategy grid...
  local label=$1 nodes=$2 data=$3 hid=$4 heads=$5 strat=$6; shift 6
  echo "### $label | gpus=$(( nodes * 4 )) $(basename $data) h$hid/$heads $strat $*  $(date +%H:%M:%S)"
  GT3D_SCRIPT=bench/bench_dist_attention.py MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 $SRUN -N $nodes \
      --ntasks-per-node 4 --gpus-per-node 4 --cpu-bind=cores bash scripts/multinode_train.sh --data $data \
      --strategy $strat --grid "$@" --hidden $hid --heads $heads --iters 10 --comm_bf16 --rs_bf16 < /dev/null 2>&1 \
      | grep "^ATTN\|OutOfMemory\|Traceback\|AssertionError" | cut -c1-300 | head -3
}

model1() {  # nodes data hidden heads args... -> prints the filtered output, returns 1 on OOM without a STEADY line
  local nodes=$1 data=$2 hid=$3 heads=$4; shift 4
  local out
  out=$(MASTER_PORT=$(( 29700 + RANDOM % 200 )) timeout 900 $SRUN -N $nodes --ntasks-per-node 4 --gpus-per-node 4 \
      --cpu-bind=cores bash scripts/multinode_train.sh --data $data --backend triton --hidden $hid --heads $heads \
      --layers 4 --timing_epochs 8 --norm ln --comm_bf16 --rs_bf16 "$@" < /dev/null 2>&1 \
      | grep -i "^STEADY\|^\[head_auto\]\|^\[nccl_tune\]\|OutOfMemory\|Traceback\|error" | grep -v warn \
      | awk '{ if ($0 ~ /^STEADY/ || $0 ~ /^\[head_auto\]/) print; else print substr($0, 1, 200) }')
  echo "$out"
  if ! grep -q "^STEADY" <<< "$out" && grep -qi "OutOfMemory" <<< "$out"; then return 1; fi
  return 0
}

model() {  # label nodes data hidden heads args...   (retries once with --checkpoint after an OOM)
  local label=$1 nodes=$2 data=$3 hid=$4 heads=$5; shift 5
  echo "### $label | gpus=$(( nodes * 4 )) $(basename $data) h$hid/$heads $*  $(date +%H:%M:%S)"
  if ! model1 $nodes $data $hid $heads "$@"; then
    echo "### $label | retry with --checkpoint (OOM without)"
    model1 $nodes $data $hid $heads "$@" --checkpoint
  fi
}

ours_attn() {  # gpus -> "strategy grid" candidates of our selector (fused 2D + head grids, P_i | 8)
  case $1 in
    4)  echo "gather 2 2|head 2 1 2|head 2 2 1";;
    8)  echo "gather 2 4|head 2 1 4|head 2 2 2";;
    16) echo "gather 4 4|head 4 1 4|head 2 2 4|head 4 2 2";;
    32) echo "gather 8 4|gather 4 8|head 8 1 4|head 4 2 4|head 4 4 2";;
    64) echo "gather 8 8|gather 16 4|head 16 1 4|head 8 2 4|head 4 4 4";;
  esac | tr '|' '\n'
}

ours_model() {  # gpus -> grid arguments of the full-model candidates (besides --head_auto)
  case $1 in
    4)  echo "2 2 --fused|2 1 2";;
    8)  echo "2 4 --fused|2 1 4";;
    16) echo "4 4 --fused|4 1 4|2 2 4";;
    32) echo "8 4 --fused|8 1 4|4 2 4";;
    64) echo "8 8 --fused|16 1 4|4 4 4";;
  esac | tr '|' '\n'
}

for nodes in "$@"; do
  gp=$(( nodes * 4 ))
  if [ "$PART" = attn ] || [ "$PART" = all ]; then
    for cfg in "$P 128 8" "$R 128 8" "$P 256 8" "$R 256 8"; do
      set -- $cfg; data=$1; hid=$2; heads=$3
      attn "GP-AG"   $nodes $data $hid $heads head $gp 1 1
      attn "GP-AG-x" $nodes $data $hid $heads gather $gp 1
      [ $gp -le $heads ] && attn "GP-A2A" $nodes $data $hid $heads head 1 1 $gp
      ours_attn $gp | while read s g; do attn "ours" $nodes $data $hid $heads $s $g; done
    done
  fi
  if [ "$PART" = model ] || [ "$PART" = all ]; then
    for cfg in "$P 128 8" "$R 128 8" "$P 256 8"; do
      set -- $cfg; data=$1; hid=$2; heads=$3
      model "GP-AG"   $nodes $data $hid $heads --grid $gp 1 1
      model "GP-AG-x" $nodes $data $hid $heads --grid $gp 1 --fused
      [ $gp -le $heads ] && model "GP-A2A" $nodes $data $hid $heads --grid 1 1 $gp
      ours_model $gp | while read g; do model "ours" $nodes $data $hid $heads --grid $g; done
      model "ours-auto" $nodes $data $hid $heads --head_auto
    done
  fi
done
echo "### done $(date +%H:%M:%S)"
