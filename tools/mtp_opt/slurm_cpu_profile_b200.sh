#!/bin/bash
# =============================================================================
# Slurm job: CPU Profile — MTP decode step breakdown on B200
# TP=2, ISL=34K, OSL=300, BS=1, BF16, MTP=3
#
# USAGE:
#   sbatch slurm_cpu_profile_b200.sh
# =============================================================================

#SBATCH --job-name=cpu_prof_mtp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --partition=b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --account=beta-users_b200
#SBATCH --qos=batch-short
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.err

set -euo pipefail

WORKSPACE_HOST="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
CONTAINER_IMAGE="${DOCKER_IMAGE:-nvcr.io#nvidia/pytorch:25.10-py3}"
HF_CACHE="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace/results/cpu_profile_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p "${WORKSPACE_HOST}/logs" "${WORKSPACE_HOST}/results"

echo "========================================================"
echo " CPU Profile — MTP Decode Step Breakdown — B200"
echo " Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo " Start: $(date)"
echo "========================================================"

srun \
    --cpu-bind=none \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${WORKSPACE_HOST}:/workspace,${HF_CACHE}:/workspace/.cache/huggingface" \
    --container-writable \
    --export="\
RESULTS_DIR=${RESULTS_DIR},\
GPU_MEM_UTIL=0.92,\
TIMEOUT=3600,\
SERVE_PROMPTS=10" \
    bash /workspace/bench_cpu_profile_inner.sh

echo ""
echo "========================================================"
echo " Done: ${SLURM_JOB_ID}  $(date)"
echo " Results: ${WORKSPACE_HOST}/results/"
echo "========================================================"
ls -lh "${WORKSPACE_HOST}/results/" 2>/dev/null || true
