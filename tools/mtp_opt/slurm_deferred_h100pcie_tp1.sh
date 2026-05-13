#!/bin/bash
#SBATCH --job-name=deferred_tp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=h100-pcie@cr+mp/h12sswnt/1gpu-16cpu-128gb
#SBATCH --account=beta-users_h100-pcie
#SBATCH --qos=batch-short
#SBATCH --output=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.out
#SBATCH --error=/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt/logs/slurm_%j.err

set -euo pipefail

WORKSPACE_HOST="/home/scratch.alichen_sw_1/workspace/exp_claude/qwen3_5/mtp_opt"
CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:25.10-py3"
HF_CACHE="/home/scratch.alichen_sw_1/workspace/exp_claude/vllm/hf_cache"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace/results/deferred_h100pcie_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p "${WORKSPACE_HOST}/logs" "${WORKSPACE_HOST}/results"

echo "========================================================"
echo " Strategy D: Deferred Draft — H100-PCIe TP=1"
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
MTP_TOKENS=1,\
TP_SIZE=1,\
SERVE_PROMPTS=15,\
GPU_MEM_UTIL=0.95,\
TIMEOUT=1800,\
EXTRA_VLLM_ARGS=--quantization fp8 --enforce-eager --max-num-seqs 4" \
    bash /workspace/bench_deferred_draft_inner.sh

echo "Done: $(date)"
