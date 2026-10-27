#!/bin/bash
# Launcher wrapper for the NVSHMEM pull-collective benchmarks: loads the NVSHMEM module (libfabric transport on
# Slingshot, CUDA VMM off), sizes the symmetric heap, then runs the usual multinode launcher.
#   srun -N <nodes> --ntasks-per-node <k> --gpus-per-node 4 bash scripts/run_nvshmem.sh <train args...>
# libfabric 1.20.1: the NVSHMEM 3.2.5 libfabric transport was built against the 1.x ABI and segfaults in fi_getinfo
# with the default libfabric 2.3.1 (2026-10-27, standalone and inside torch alike); with 1.20.1 it initialises.
module load libfabric/1.20.1 nvshmem/3.2.5-1 2>/dev/null
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-libfabric} NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4294967296}
exec bash /pscratch/sd/c/cunyang/gnn/gt/gt3d/scripts/multinode_train.sh "$@"
