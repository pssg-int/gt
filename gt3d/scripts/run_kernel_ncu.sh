#!/bin/bash
# Nsight Compute on the attention kernel passes (bench/kernel_ncu.py) on one A100-40GB of a whole node.
# Run inside an allocation: srun -N 1 ... bash scripts/run_kernel_ncu.sh  (dcgmi pause / resume around ncu)
set -u
cd /pscratch/sd/c/cunyang/gnn/gt/gt3d
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export PYTHONPATH=/pscratch/sd/c/cunyang/gnn/gt/gt3d:${PYTHONPATH:-}
NCU=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu
SECTIONS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/sections   # stock sections (the per-user copy is incomplete)
OUT=bench/kernel_ncu
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
dcgmi profile --pause
CUDA_VISIBLE_DEVICES=0 $NCU --section-folder $SECTIONS --profile-from-start off --kernel-name regex:_attn_ \
     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats \
     --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_requests_srcunit_tex_op_red.sum,lts__t_requests_srcunit_tex_op_atom.sum \
     -f -o $OUT python bench/kernel_ncu.py
rc=$?
dcgmi profile --resume
echo "ncu exit $rc"
$NCU --import $OUT.ncu-rep --csv --page details > $OUT.csv && echo "csv written $OUT.csv"
