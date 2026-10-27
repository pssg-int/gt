#!/bin/bash
# Run an arbitrary command inside the plexus_env on a compute node (via srun --jobid ... bash scripts/run_on_node.sh <cmd...>)
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export DGL_DISABLE_GRAPHBOLT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTHONPATH=/pscratch/sd/c/cunyang/gnn/gt/gt3d:/pscratch/sd/c/cunyang/gnn/plexus:/pscratch/sd/c/cunyang/gnn/baseline/Exphormer:$PYTHONPATH
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
echo "[run_on_node] host=$(hostname) start=$(date) cmd: $*"
"$@"
echo "[run_on_node] exit=$? end=$(date)"
