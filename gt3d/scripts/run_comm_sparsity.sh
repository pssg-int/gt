#!/bin/bash
# Communication sparsity of the gather layouts on products, Reddit and arxiv (CPU node, ~20 min).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d || exit 1
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export OMP_NUM_THREADS=32
export PYTHONPATH=/pscratch/sd/c/cunyang/gnn/gt/gt3d:/pscratch/sd/c/cunyang/gnn/plexus:/pscratch/sd/c/cunyang/gnn/baseline/Exphormer:$PYTHONPATH
python bench/comm_sparsity.py --arxiv --name arxiv
python bench/comm_sparsity.py --data /pscratch/sd/c/cunyang/gnn/gt/data/reddit_p16 --name reddit
python bench/comm_sparsity.py --data /pscratch/sd/c/cunyang/gnn/gt/data/products_p16b --name products
