#!/bin/bash
# =============================================================================
# Slurm job: MTP1 optimization benchmark — Qwen3.5-35B-A3B on B200
# Tests: no_mtp, mtp1_base, mtp1_o1, mtp1_o2, mtp1_o1_o2, gsm8k
# =============================================================================

#SBATCH --job-name=mtp1_opts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --partition=b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --account=beta-users_b200
#SBATCH --qos=batch-short
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.err

set -euo pipefail

WORKSPACE_HOST="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
CONTAINER_IMAGE="${DOCKER_IMAGE:-nvcr.io#nvidia/pytorch:25.10-py3}"
HF_CACHE="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"

RUN_CONFIGS="${RUN_CONFIGS:-all}"
SERVE_PROMPTS="${SERVE_PROMPTS:-15}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
TIMEOUT="${TIMEOUT:-1800}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace/results/mtp1_opts_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p "${WORKSPACE_HOST}/logs" "${WORKSPACE_HOST}/results"

echo "========================================================"
echo " MTP1 Optimization Bench — B200"
echo " Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo " Configs: ${RUN_CONFIGS}"
echo " Start: $(date)"
echo "========================================================"

srun \
    --cpu-bind=none \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${WORKSPACE_HOST}:/workspace,${HF_CACHE}:/workspace/.cache/huggingface" \
    --container-writable \
    --export="\
RUN_CONFIGS=${RUN_CONFIGS},\
SERVE_PROMPTS=${SERVE_PROMPTS},\
GPU_MEM_UTIL=${GPU_MEM_UTIL},\
TIMEOUT=${TIMEOUT},\
RESULTS_DIR=${RESULTS_DIR}" \
    bash /workspace/bench_mtp1_opts_inner.sh

echo ""
echo "========================================================"
echo " Done: ${SLURM_JOB_ID}  $(date)"
echo " Results: ${WORKSPACE_HOST}/results/"
echo "========================================================"
