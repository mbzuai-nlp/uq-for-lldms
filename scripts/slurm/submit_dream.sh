#!/bin/bash
# SLURM submit script for Dream-7B polygraph_eval runs on the CIAI cluster.
#
# Usage:
#   sbatch scripts/slurm/submit_dream.sh DATASET [EXTRA_HYDRA_ARGS...]
#
# Examples:
#   sbatch scripts/slurm/submit_dream.sh mmlu_simple_instruct subsample_eval_dataset=200
#   sbatch scripts/slurm/submit_dream.sh xsum_simple_instruct '+max_input_tokens=2048'
#
# DATASET is the suffix matching examples/configs/polygraph_eval_<DATASET>.yaml
# (e.g. "mmlu_simple_instruct", "wmt19_deen_minimal").

#SBATCH -J dream-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/artem.shelmanov/vlad/uncertainty_dllm/scripts/slurm/logs/%x-%j.out
#SBATCH -e /home/artem.shelmanov/vlad/uncertainty_dllm/scripts/slurm/logs/%x-%j.err

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "ERROR: dataset slug required (e.g. mmlu_simple_instruct)" >&2
  exit 64
fi
shift
EXTRA=("$@")

PROJECT=/home/artem.shelmanov/vlad/uncertainty_dllm
HYDRA_CFG=$PROJECT/examples/configs/polygraph_eval_${DATASET}.yaml
if [[ ! -f $HYDRA_CFG ]]; then
  echo "ERROR: config not found: $HYDRA_CFG" >&2
  exit 65
fi

source /home/artem.shelmanov/miniconda3/etc/profile.d/conda.sh
conda activate uncertainty_dllm

# Ignore user-local site-packages at ~/.local/ — they contain a stale
# transformers/torch stack that is ABI-incompatible with the env's
# torch+nccl pair (causes ncclCommWindowDeregister undefined-symbol
# ImportError on import torch).
export PYTHONNOUSERSITE=1

# CUDA visibility — SLURM already exports this when --gres=gpu:N is set.
echo "[submit] node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} dataset=$DATASET"
echo "[submit] commit=$(cd $PROJECT && git rev-parse --short HEAD)"
echo "[submit] extra=${EXTRA[*]:-}"
echo "[submit] start: $(date -Iseconds)"

CACHE_PATH=$PROJECT/workdir/ciai_${SLURM_JOB_ID}_${DATASET}

cd $PROJECT
env HYDRA_CONFIG=$HYDRA_CFG python polygraph_eval \
  batch_size=1 \
  model.path=Dream-org/Dream-v0-Instruct-7B \
  model.path_to_load_script=$PROJECT/llada/load_dream_cached.py \
  model.type=DiffusionLLM \
  '+model.device=cuda' \
  '+model.shift_logits=true' \
  '+model.use_cache=true' \
  '+model.mask_id=151666' \
  estimators=all_estimators_llada_no_train_diffusion \
  stat_calculators=new_calculators_llada \
  cache_path=$CACHE_PATH \
  '+log_time=true' \
  "${EXTRA[@]}"

echo "[submit] end: $(date -Iseconds) rc=$?"
