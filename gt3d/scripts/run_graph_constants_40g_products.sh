#!/bin/bash
# products on an A100-40GB node: the 1x1 block with 8 heads does not fit, so the 1x1 grid runs the smaller head configurations
D=/pscratch/sd/c/cunyang/gnn/gt/data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/graph_constants.py --data $D/products_p16b --name products_40g --configs 4x32,8x32,2x40,1x32,2x32,1x40,2x20 --blocks 2,4,8,16,32,64
python bench/graph_constants.py --data $D/products_p16b --name products_1x1_40g --configs 4x32,2x32,1x32,2x40,1x40,2x20 --grids 1x1
