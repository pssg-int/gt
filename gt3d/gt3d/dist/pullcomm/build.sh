#!/bin/bash
# Build libpullcomm.so against the system NVSHMEM (built with CUDA 12.4) with the matching CUDA 12.4 nvcc and a SHARED
# cudart, so that inside a torch process the library binds to the already-loaded libcudart.so.12 instead of carrying
# a second runtime (a static or 13.x cudart next to torch's 12.8 crashed nvshmemx_init_attr, 2026-10-27).
#   bash gt3d/dist/pullcomm/build.sh
set -e
cd "$(dirname "$0")"
NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc   # a CUDA 12.x nvcc: its cudart soname (libcudart.so.12) is the one torch already loads
: "${NVSHMEM_HOME:=/global/common/software/nersc9/nvshmem/3.2.5-1-25.03}"
$NVCC -O3 -std=c++17 -arch=sm_80 -rdc=true -shared -Xcompiler -fPIC -cudart shared \
     -I"$NVSHMEM_HOME/include" pullcomm.cu -o libpullcomm.so \
     -L"$NVSHMEM_HOME/lib" -lnvshmem_host -lnvshmem_device -lcuda \
     -Xlinker -rpath -Xlinker "$NVSHMEM_HOME/lib"
echo "built $(pwd)/libpullcomm.so"
