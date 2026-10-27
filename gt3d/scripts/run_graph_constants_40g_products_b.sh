#!/bin/bash
D=/pscratch/sd/c/cunyang/gnn/gt/data
python bench/graph_constants.py --data $D/products_p16b --name products_1x1_40g --configs 4x32,2x32,1x32,2x40,1x40,2x20 --grids 1x1
