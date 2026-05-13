#!/bin/bash
# =============================================================================
# P0 CPU Overhead Optimization — H100 Validation
# Qwen3.5-35B-A3B  TP=2  MTP=3  ISL=34K  OSL=300  BS=1  BF16
#
# USAGE:
#   sbatch slurm_p0_opt_h100.sh
#   sbatch --export=SERVE_PROMPTS=20 slurm_p0_opt_h100.sh
# =============================================================================

#SBATCH --job-name=p0_opt_mtp_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --partition=h100-80gb-hbm3@ts6/viking@dvt/8gpu-224cpu-2048gb
#SBATCH --account=beta-users_h100-80gb-hbm3
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_p0_opt_h100_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_p0_opt_h100_%j.err

set -euo pipefail

WORKSPACE_HOST="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
HF_CACHE="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"
CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:25.10-py3"

SERVE_PROMPTS="${SERVE_PROMPTS:-15}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
TIMEOUT="${TIMEOUT:-1800}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace/results/p0_opt_h100_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p "${WORKSPACE_HOST}/logs" "${WORKSPACE_HOST}/results"

echo "========================================================"
echo " P0 CPU Opt MTP=3 H100 | Job: ${SLURM_JOB_ID}"
echo " Node: $(hostname)"
echo " Start: $(date)"
echo "========================================================"

srun \
    --cpu-bind=none \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${WORKSPACE_HOST}:/workspace,${HF_CACHE}:/workspace/.cache/huggingface" \
    --container-writable \
    --export="\
SERVE_PROMPTS=${SERVE_PROMPTS},\
GPU_MEM_UTIL=${GPU_MEM_UTIL},\
TIMEOUT=${TIMEOUT},\
RESULTS_DIR=${RESULTS_DIR}" \
    bash /workspace/bench_p0_opt_inner.sh

echo ""
echo "========================================================"
echo " Done: ${SLURM_JOB_ID}  $(date)"
echo " Results: ${WORKSPACE_HOST}/results/"
echo "========================================================"
ls -lh "${WORKSPACE_HOST}/results/" 2>/dev/null || true
