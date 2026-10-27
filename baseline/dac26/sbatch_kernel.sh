#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH -A gen0034_g
#SBATCH -J dac26-kernel
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26/logs/kernel_%j.log
# One SGA attention (h = 8 x 16, fp32) fwd + bwd on the whole graph, one A100-80GB: DGL 2.3 bsddmm -> softmax -> bspmm
# (the DAC'26 stack) vs our Triton kernel (score dot, norm softmax), same edges and inputs; ours is checked against the
# DGL output and gradients first.
cd /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26
echo "### kernel job=$SLURM_JOB_ID $(date)"
for g in proteins reddit products; do
  MASTER_PORT=29651 timeout 600 srun -N 1 -n 1 --gpus-per-node 4 bash launch.sh dac26_sga.py --graph $g --kernel \
      --dump kernel_ref/$g.pt < /dev/null 2>&1 | grep "^RESULT\|Error\|Traceback" | cut -c1-300
done
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
srun -N 1 -n 1 --gpus-per-node 4 bash -c "source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate; \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u bench/bench_sga_kernel.py --graphs proteins reddit products \
    --check_dir /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26/kernel_ref" < /dev/null 2>&1 | grep "^RESULT\|^CHECK\|Error\|Traceback" | cut -c1-300
echo "### done $(date)"
