#!/bin/bash
# Re-measure the per-graph kernel calibration tables on an A100-40GB node (outputs *_40g.json; originals untouched)
D=/pscratch/sd/c/cunyang/gnn/gt/data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/graph_constants.py --data $D/products_p16b --name products_40g --configs 4x32,8x32,2x40,1x32,2x32,1x40,2x20 --blocks 1,2,4,8,16,32,64
python bench/graph_constants.py --data $D/reddit_p16 --name reddit_40g --configs 4x32,8x32,1x32,2x32 --blocks 1,2,4,8,16,32,64
python bench/graph_constants.py --data $D/products_p16b --name products_hs_40g --configs 1x32,2x32,1x40,2x20 --grids 2x2,4x4,8x8,16x4,4x16
python bench/graph_constants.py --data $D/reddit_p16 --name reddit_hs_40g --configs 1x32,2x32 --grids 2x2,4x4,8x8,16x4
