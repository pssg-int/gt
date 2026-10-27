#!/bin/bash
D=/pscratch/sd/c/cunyang/gnn/gt/data
python bench/graph_constants.py --data $D/products_p16b --name products_40g --configs 4x32,8x32,2x40,1x32,2x32,1x40,2x20 --blocks 2,4,8,16,32,64
