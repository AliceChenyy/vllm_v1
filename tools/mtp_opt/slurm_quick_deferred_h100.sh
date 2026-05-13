#!/bin/bash
#SBATCH --job-name=quick_defer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --partition=h100-80gb-hbm3@ts6/mg62g4100/1gpu-32cpu-256gb
#SBATCH --account=beta-users_h100-80gb-hbm3
#SBATCH --qos=batch-short
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.err

set -euo pipefail
WS="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
HF="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"
mkdir -p "${WS}/logs" "${WS}/results"

srun --cpu-bind=none \
  --container-image="nvcr.io#nvidia/pytorch:25.10-py3" \
  --container-mounts="${WS}:/workspace,${HF}:/workspace/.cache/huggingface" \
  --container-writable \
  --export="TP_SIZE=1,EXTRA_VLLM_ARGS=--enforce-eager" \
  bash /workspace/quick_deferred_test.sh
