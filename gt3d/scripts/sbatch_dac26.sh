#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --gpus-per-node 4
#SBATCH -A gen0034_g
#SBATCH -J gt3d-dac26
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_dac26_%j.log
# Batch wrapper of scripts/run_dac26_strategies.sh (DAC'26 GP-AG / GP-A2A on our kernel vs our layouts):
#   sbatch -N 4  -t 01:15:00 --export=ALL,PART=attn,NODES="1 2 4"  scripts/sbatch_dac26.sh
#   sbatch -N 16 -t 01:30:00 --export=ALL,PART=all,NODES="16"      scripts/sbatch_dac26.sh
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
echo "### sbatch_dac26 job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES part=$PART steps=[$NODES] $(date)"
bash scripts/run_dac26_strategies.sh - $PART $NODES
