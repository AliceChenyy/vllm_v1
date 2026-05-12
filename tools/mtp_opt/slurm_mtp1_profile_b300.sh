#!/bin/bash
# =============================================================================
# Slurm job: MTP1 decode step profiling — Qwen3.5-35B-A3B on B300
# Profiles no_mtp vs mtp1 to identify MTP overhead source
# =============================================================================

#SBATCH --job-name=mtp1_prof
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --partition=b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb
#SBATCH --account=beta-users_b300
#SBATCH --qos=batch-short
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.err

set -euo pipefail

WORKSPACE_HOST="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
CONTAINER_IMAGE="${DOCKER_IMAGE:-nvcr.io#nvidia/pytorch:25.10-py3}"
HF_CACHE="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace/results/mtp1_profile_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p "${WORKSPACE_HOST}/logs" "${WORKSPACE_HOST}/results"

echo "========================================================"
echo " MTP1 Decode Step Profiling — B300"
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
SERVE_PROMPTS=${SERVE_PROMPTS:-10},\
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92},\
TIMEOUT=${TIMEOUT:-3600}" \
    bash /workspace/bench_mtp1_profile_inner.sh

echo ""
echo "========================================================"
echo " Done: ${SLURM_JOB_ID}  $(date)"
echo " Results: ${WORKSPACE_HOST}/results/"
echo "========================================================"
