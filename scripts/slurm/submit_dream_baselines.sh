#!/bin/bash
# SLURM submit script for Dream-7B baseline-estimator polygraph_eval runs on
# CIAI. Uses baseline_estimators_llada_no_train + new_calculators_llada_no_train
# (vs the diffusion-trace estimators in submit_dream.sh).
#
# Usage:
#   sbatch scripts/slurm/submit_dream_baselines.sh DATASET [EXTRA_HYDRA_ARGS...]
#
# Example:
#   sbatch scripts/slurm/submit_dream_baselines.sh coqa_simple_instruct
#
# DATASET is the suffix matching examples/configs/polygraph_eval_<DATASET>.yaml.

#SBATCH -J dream-baselines
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
  echo "ERROR: dataset slug required (e.g. coqa_simple_instruct)" >&2
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

# Stale ~/.local/ has ABI-incompatible torch+transformers.
export PYTHONNOUSERSITE=1

# Defragment the CUDA caching allocator: stat-calc loops over transient
# logits/attention tensors fragment the pool fast on long-prompt datasets.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[submit] node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} dataset=$DATASET"
echo "[submit] commit=$(cd $PROJECT && git rev-parse --short HEAD)"
echo "[submit] extra=${EXTRA[*]:-}"
echo "[submit] start: $(date -Iseconds)"

CACHE_PATH=$PROJECT/workdir/ciai_${SLURM_JOB_ID}_${DATASET}_baselines

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
  estimators=baseline_estimators_llada_no_train \
  stat_calculators=new_calculators_llada_no_train \
  cache_path=$CACHE_PATH \
  '+log_time=true' \
  '+deberta_batch_size=4' \
  "${EXTRA[@]}"

echo "[submit] end: $(date -Iseconds) rc=$?"
