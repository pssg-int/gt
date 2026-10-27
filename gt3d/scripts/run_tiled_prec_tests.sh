#!/bin/bash
# Correctness of the tensor-core precision modes of the tiled kernels (vs the per-row kernels, 1e-4 tolerance).
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
for prec in tf32x2 tf32; do
  echo "### GT3D_TILED_PREC=$prec $(date +%H:%M:%S)"
  GT3D_TILED_PREC=$prec python tests/test_triton_tiled.py 2>&1 | tail -30
done
echo "### done $(date +%H:%M:%S)"
