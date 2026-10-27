#!/bin/bash
# One task per GPU under srun: environment of the DGL re-implementation (torch 2.3.0+cu121, DGL 2.3.0, the paper's
# stack) + torch.distributed rendezvous from the Slurm step.
#   srun -N 1 --ntasks-per-node 4 --gpus-per-node 4 bash launch.sh dac26_sga.py --graph products --strategy gp_ag
source /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26_env/bin/activate
SP=/pscratch/sd/c/cunyang/gnn/gt/baseline/dac26_env/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export DGL_DISABLE_GRAPHBOLT=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# multi-node steps: the NERSC libfabric (Slingshot) plugin for the NCCL bundled with torch
if [ "${SLURM_STEP_NUM_NODES:-1}" -gt 1 ]; then
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-ofi} NCCL_SOCKET_IFNAME=hsn NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
  export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.24.3/plugin/lib:$LD_LIBRARY_PATH
fi
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-$SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29650}
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
cd /pscratch/sd/c/cunyang/gnn/gt/baseline/dac26
exec python -u "$@"
