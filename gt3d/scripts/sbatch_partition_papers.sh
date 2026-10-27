#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node 1
#SBATCH -t 03:00:00
#SBATCH -A m2404
#SBATCH -J gt3d-partition-papers
#SBATCH -o /pscratch/sd/c/cunyang/gnn/gt/gt3d/logs/sbatch_partition_papers_%j.log
# 2D partition of ogbn-papers100M (single-permutation Plexus processed file, 126 GB) into 32x32 blocks.
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export PYTHONPATH=/pscratch/sd/c/cunyang/gnn/gt/gt3d:$PYTHONPATH
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
free -g | head -2
python -u -m gt3d.data.partition --src /pscratch/sd/c/cunyang/gnn/plexus/dataset/papers/undirected_sym/processed/processed_papers.pt \
    --out /pscratch/sd/c/cunyang/gnn/gt/data/papers_p32 --p_part 32
du -sh /pscratch/sd/c/cunyang/gnn/gt/data/papers_p32
