#!/bin/bash
# Multi-node launcher: one task per GPU via srun; torch.distributed reads MASTER_ADDR/PORT/RANK/WORLD_SIZE.
#   srun --jobid=<id> -N <nodes> --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh <train args...>
# or inside an sbatch/salloc shell:  srun -N $SLURM_NNODES --ntasks-per-node 4 --gpus-per-node 4 bash scripts/multinode_train.sh ...
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
module load nccl/2.24.3 2>/dev/null
export DGL_DISABLE_GRAPHBOLT=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# expandable segments: no reserved-but-unallocated fragmentation (papers100M 64 GPUs: 7 GB reserved on top of 30 GB allocated)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=/pscratch/sd/c/cunyang/gnn/gt/gt3d:/pscratch/sd/c/cunyang/gnn/plexus:$PYTHONPATH
export NCCL_SOCKET_IFNAME=hsn NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
# NCCL point-to-point over the network (ring attention hops): 4 channels per net peer + 1 MiB chunks; with the defaults a
# [2, chunk, H d] hop ran at ~20 GB/s per rank whichever direction split was used (RESULTS.md 2026-10-13, bidirectional
# ring).  Collectives (all-gather / reduce-scatter / all-to-all) are not affected by these two variables.
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-4} NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-1048576}
if [ "${GT3D_FI_DEFAULTS:-0}" = "1" ]; then                     # diagnosis: libfabric CXI defaults instead of the eager / overflow overrides
  unset FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN FI_CXI_OFLOW_BUF_SIZE FI_CXI_OFLOW_BUF_COUNT
else
  export FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_OFLOW_BUF_SIZE=1073741824 FI_CXI_OFLOW_BUF_COUNT=1
fi
# the step may run on a subset of the job nodes: the master must be the FIRST NODE OF THE STEP
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-$SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29600}
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
[ "$RANK" = "0" ] && echo "[multinode] nodes=$SLURM_JOB_NUM_NODES world=$WORLD_SIZE master=$MASTER_ADDR start=$(date) args: $*"
exec python -u ${GT3D_SCRIPT:-gt3d/train/train_dist.py} "$@"
