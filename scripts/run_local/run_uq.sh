#!/bin/bash
# Run the full UQ benchmark (all_estimators_llada_complete) on a local multi-GPU machine (no Slurm).
#
# Mirrors the settings of the cluster runs used in the paper
# (estimators=all_estimators_llada_complete, stat_calculators=new_calculators_llada_new_baselines),
# including the per-dataset training-subsample sizes, target metrics and metric_thr overrides.
#
# usage: scripts/run_local/run_uq.sh MODEL DATASET GPU [extra hydra overrides...]
#   MODEL:   llada | dream | refusion | diffgemma (run from the feat/diffusiongemma-uq checkout)
#   DATASET: triviaqa | coqa | mmlu | gsm8k | samsum | xsum | wmt14 | wmt19
#   GPU:     CUDA device id(s), e.g. 0 or 0,1
# env: N_EVAL (default 2000), ESTIMATORS / CALCULATORS (default all_estimators_llada_complete /
#      new_calculators_llada_new_baselines; *_diffusiongemma_complete for diffgemma), OUT (output root),
#      CONDA_ENV (default dllm; diffgemma for diffgemma),
#      TRAIN_N_OVERRIDE (override the per-dataset training subsample, e.g. 20 for a smoke test)
set -euo pipefail

MODEL=${1:?model}; DATASET=${2:?dataset}; GPU=${3:?gpu}; shift 3
EXTRA=("$@")

PROJECT=$(cd "$(dirname "$0")/../.." && pwd)
N_EVAL=${N_EVAL:-2000}
OUT=${OUT:-$PROJECT/workdir/output_42_complete}
if [[ $MODEL == diffgemma ]]; then
  ESTIMATORS=${ESTIMATORS:-estimators_diffusiongemma_complete}
  CALCULATORS=${CALCULATORS:-calculators_diffusiongemma_complete}
  CONDA_ENV=${CONDA_ENV:-diffgemma}
fi
ESTIMATORS=${ESTIMATORS:-all_estimators_llada_complete}
CALCULATORS=${CALCULATORS:-new_calculators_llada_new_baselines}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-dllm}"

# Indices of the estimators named $1 (optionally only those whose cfg has key $2) in $ESTIMATORS.
est_idx() {
  python - "$PROJECT/examples/configs/estimators/$ESTIMATORS.yaml" "$@" <<'PY'
import sys, yaml
c, name = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else None
print(" ".join(str(i) for i, e in enumerate(c) if e.get("name") == name and (key is None or key in (e.get("cfg") or {}))))
PY
}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
# datasets==3.6 (pinned, as on the cluster) cannot read caches written by datasets 4.x from other envs here.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets_v3}

case $MODEL in
  llada)
    N_LAYERS=32
    MODEL_ARGS=(model.path=GSAI-ML/LLaDA-1.5 model.path_to_load_script="$PROJECT/llada/load_llada.py"
                model.type=DiffusionLLM +model.device=cuda) ;;
  dream)
    N_LAYERS=28
    MODEL_ARGS=(model.path=Dream-org/Dream-v0-Instruct-7B model.path_to_load_script="$PROJECT/dream/load_dream_cached.py"
                model.type=DiffusionLLM +model.device=cuda +model.shift_logits=true +model.use_cache=true
                +model.mask_id=151666) ;;
  refusion)
    N_LAYERS=36
    MODEL_ARGS=(model=refusion model.path=GSAI-ML/ReFusion model.path_to_load_script="$PROJECT/refusion/load_refusion.py"
                model.type=DiffusionLLM +model.device=cuda model.mask_id=151670 model.use_cache=true
                ++trust_remote_code=false) ;;
  diffgemma)
    N_LAYERS=30
    MODEL_ARGS=(model=diffusiongemma model.path_to_load_script="$PROJECT/llada/load_diffusiongemma.py"
                ignore_exceptions=true) ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
# The complete configs hard-code ReFusion's depth (Qwen3-8B, 36 layers) for AttentionScore (middle
# layer) and RAUQ; RAUQ crashes on any other depth.
for i in $(est_idx AttentionScore layer); do MODEL_ARGS+=("estimators.$i.cfg.layer=$((N_LAYERS / 2))"); done
for i in $(est_idx RAUQ n_layers); do MODEL_ARGS+=("estimators.$i.cfg.n_layers=$N_LAYERS"); done

# Per-dataset settings, copied from the cluster ReFusion runs.
TRAIN_N=1000
DS_ARGS=()
case $DATASET in
  triviaqa) CFG=triviaqa_simple_instruct;    METRIC=AlignScore ;;
  coqa)     CFG=coqa_simple_instruct;        METRIC=AlignScore ;;
  samsum)   CFG=samsum_simple_instruct;      METRIC=AlignScore ;;
  mmlu)     CFG=mmlu_simple_instruct;        METRIC=Accuracy ;;
  gsm8k)    CFG=gsm8k_simple_instruct;       METRIC=Accuracy;   TRAIN_N=500 ;;
  xsum)     CFG=xsum_simple_instruct;        METRIC=AlignScore; TRAIN_N=500
            DS_ARGS=(+stat_calculators.5.cfg.mc_batch_size=4 +stat_calculators.7.cfg.mc_batch_size=4) ;;
  wmt14|wmt19)
            [[ $DATASET == wmt14 ]] && CFG=wmt14_fren_simple_instruct || CFG=wmt19_deen_simple_instruct
            METRIC=Comet
            # AlignScore is where every exit=134 crash surfaces (async detector of GPU hangs); skip it for wmt14
            [[ $DATASET == wmt14 ]] && export DISABLE_ALIGNSCORE=1 || true
            # Supervised estimators binarise the target at metric_thr; COMET needs 0.85 instead of 0.3
            # (as in the cluster runs: MIND, Sheeps, SATRMD, LookBackLens, tracedet_timehalu, dynhd).
            for n in MIND Sheeps SATRMD LookBackLens methods.tracedet_timehalu methods.dynhd; do
              for i in $(est_idx $n metric_thr); do DS_ARGS+=("estimators.$i.cfg.metric_thr=0.85"); done
            done ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac
TRAIN_N=${TRAIN_N_OVERRIDE:-$TRAIN_N}

cd "$PROJECT"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/${MODEL}_${DATASET}_$(date +%Y%m%d-%H%M%S).log"
echo "log: $LOG"

HYDRA_CONFIG="$PROJECT/examples/configs/polygraph_eval_${CFG}.yaml" \
  python -u polygraph_eval \
    batch_size=1 \
    size=10000 \
    subsample_eval_dataset="$N_EVAL" \
    estimators="$ESTIMATORS" \
    stat_calculators="$CALCULATORS" \
    stat_calculators.14.cfg.subsample_train_dataset=$TRAIN_N \
    stat_calculators.14.cfg.subsample_background_train_dataset=$TRAIN_N \
    stat_calculators.14.cfg.target_metric=$METRIC \
    "${MODEL_ARGS[@]}" \
    "${DS_ARGS[@]}" \
    cache_path="$OUT/$MODEL" \
    +attention_device=cpu \
    +log_time=True \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
