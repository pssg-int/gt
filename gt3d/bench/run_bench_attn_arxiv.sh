#!/bin/bash
# Single-GPU attention layer timings on ogbn-arxiv (official scatter vs Backend A vs Triton prototype).
set -x
python -u bench/bench_attention_single_gpu.py --heads 2 --head_dim 40
python -u bench/bench_attention_single_gpu.py --heads 2 --head_dim 40 --global_token
python -u bench/bench_attention_single_gpu.py --heads 8 --head_dim 32
python -u bench/bench_attention_single_gpu.py --heads 8 --head_dim 32 --global_token
python -u bench/bench_attention_single_gpu.py --heads 8 --head_dim 32 --dtype bf16 --impl backend_a,triton
