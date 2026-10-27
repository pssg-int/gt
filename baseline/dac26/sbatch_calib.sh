#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J dac26-calib
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26/logs/calib_%j.log
# Calibration of the DGL re-implementation of DAC'26 against the paper on one 4 x A100-80GB node:
#   1. parity tests (GP-AG / GP-A2A vs single GPU, fp64, p = 2, 4)
#   2. Table 2: MM / SpMM / SDDMM (single-head d = 128 and batched 8 heads) on one GPU, proteins / products / Reddit
#   3. Fig. 3: per-iteration time on 1 / 2 / 4 GPUs, both strategies (hidden 128, 8 heads, 3 layers, fp32)
cd /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26
S="srun -N 1 --gpus-per-node 4 --cpu-bind=cores"
step() {  # ntasks args...
  local nt=$1; shift
  MASTER_PORT=$(( 29650 + RANDOM % 200 )) timeout 600 $S --ntasks-per-node $nt bash launch.sh "$@" < /dev/null 2>&1 \
      | grep "^TEST\|^RESULT\|Error\|error\|Traceback" | grep -v "Warning" | cut -c1-300 | head -6
}
echo "### calib job=$SLURM_JOB_ID $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
for p in 2 4; do for s in gp_ag gp_a2a; do step $p test_dac26.py --strategy $s; done; done
for g in proteins reddit products; do step 1 dac26_sga.py --graph $g --micro; done
for g in proteins reddit products; do
  step 1 dac26_sga.py --graph $g
  for p in 2 4; do for s in gp_ag gp_a2a; do step $p dac26_sga.py --graph $g --strategy $s; done; done
done
echo "### done $(date)"
