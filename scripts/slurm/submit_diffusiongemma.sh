#!/bin/bash
# SLURM submit script for DiffusionGemma (google/diffusiongemma-26B-A4B-it)
# polygraph_eval runs. Sharded across 2x A100-40GB via device_map=auto (~25.9 GB/GPU).
# See docs/diffusiongemma/04_infra.md.
#
# Usage:
#   sbatch scripts/slurm/submit_diffusiongemma.sh DATASET [EXTRA_HYDRA_ARGS...]
#
# Examples:
#   sbatch scripts/slurm/submit_diffusiongemma.sh samsum_simple_instruct subsample_eval_dataset=100
#   sbatch scripts/slurm/submit_diffusiongemma.sh wmt19_deen_simple_instruct '+attention_device=cpu'
#
# DATASET is the suffix matching examples/configs/polygraph_eval_<DATASET>.yaml
# (e.g. "samsum_simple_instruct", "wmt19_deen_simple_instruct", "triviaqa_simple_instruct").

#SBATCH -J diffgemma-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p long
#SBATCH --gres=gpu:a100-sxm4-40gb:4   # CIAI: 4x A100-40GB (26B MoE ~51.7GB shards ~13GB/GPU + NLI headroom)
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -o /home/artem.shelmanov/vlad/uncertainty_dllm/scripts/slurm/logs/%x-%j.out
#SBATCH -e /home/artem.shelmanov/vlad/uncertainty_dllm/scripts/slurm/logs/%x-%j.err

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "ERROR: dataset slug required (e.g. samsum_simple_instruct)" >&2
  exit 64
fi
shift
EXTRA=("$@")

PROJECT="${DG_PROJECT:-/home/artem.shelmanov/vlad/uncertainty_dllm}"
HYDRA_CFG=$PROJECT/examples/configs/polygraph_eval_${DATASET}.yaml
if [[ ! -f $HYDRA_CFG ]]; then
  echo "ERROR: config not found: $HYDRA_CFG" >&2
  exit 65
fi

# --- Isolated DiffusionGemma env (NOT the working uncertainty_dllm env). ---
# See docs/diffusiongemma/04_infra.md: torch(cu124/cu128) + lm-polygraph@supervised_methods
# + transformers==5.14 (floor 5.11) + diffusers==0.39 (--no-deps).
DG_ROOT="${DG_ROOT:-/lustre/$USER/diffgemma}"
ENV_PREFIX="${DG_ENV_PREFIX:-$DG_ROOT/envs/diffgemma}"

source /home/artem.shelmanov/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

# Ignore user-local site-packages (~/.local): they carry a stale torch/transformers
# stack that is ABI-incompatible with the isolated env.
export PYTHONNOUSERSITE=1

# HF cache on the persistent Lustre path (26B checkpoint — do not re-download per job).
export HF_HOME="${HF_HOME:-$DG_ROOT/hf_cache}"
mkdir -p "$HF_HOME"

# Reduce fragmentation for the sharded 26B model.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[submit] node=$(hostname) job=${SLURM_JOB_ID:-?} gpus=${CUDA_VISIBLE_DEVICES:-?} dataset=$DATASET"
echo "[submit] env_prefix=$ENV_PREFIX hf_home=$HF_HOME"
echo "[submit] commit=$(cd "$PROJECT" && git rev-parse --short HEAD)"
echo "[submit] extra=${EXTRA[*]:-}"
echo "[submit] start: $(date -Iseconds)"

CACHE_PATH=$PROJECT/workdir/dg_${SLURM_JOB_ID:-local}_${DATASET}

cd "$PROJECT"
env HYDRA_CONFIG=$HYDRA_CFG python polygraph_eval \
  batch_size=1 \
  model=diffusiongemma \
  estimators=estimators_diffusiongemma \
  stat_calculators=calculators_diffusiongemma \
  cache_path=$CACHE_PATH \
  '+log_time=true' \
  "${EXTRA[@]}"

echo "[submit] end: $(date -Iseconds) rc=$?"
