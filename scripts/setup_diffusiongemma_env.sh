#!/usr/bin/env bash
# Create an ISOLATED python env for DiffusionGemma UQ, without touching the
# working `uncertainty_dllm` env (transformers 5.8.1, no diffusers).
#
# DiffusionGemma (google/diffusiongemma-26B-A4B-it) needs transformers >= 5.11
# (5.14 pinned) + diffusers, which are incompatible with the working env.
#
# Usage:
#   DG_ROOT=/lustre/$USER/diffgemma bash scripts/setup_diffusiongemma_env.sh          # A100 (cu124)
#   DG_ROOT=/workspace/diffgemma GPU=blackwell bash scripts/setup_diffusiongemma_env.sh  # RunPod (cu128)
set -euo pipefail

DG_ROOT="${DG_ROOT:?set DG_ROOT to a persistent path, e.g. /lustre/$USER/diffgemma or /workspace/diffgemma}"
GPU="${GPU:-a100}"                       # a100 -> cu124 ; blackwell -> cu128
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_PREFIX="$DG_ROOT/envs/diffgemma"
export HF_HOME="$DG_ROOT/hf_cache"
# HUGGINGFACE_HUB_CACHE overrides HF_HOME for the hub dir; site profiles may point it
# at node-local /tmp, so force it to shared storage explicitly (compute nodes need it).
export HUGGINGFACE_HUB_CACHE="$DG_ROOT/hf_cache/hub"
mkdir -p "$DG_ROOT/envs" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda create -y -p "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1                 # ignore ABI-incompatible ~/.local (repo convention)

# 1) torch FIRST, matched to the cluster CUDA
# torchvision is REQUIRED: DiffusionGemma's checkpoint is Gemma4-multimodal, so
# AutoProcessor -> Gemma4Processor -> image_processing_gemma4 imports torchvision.
# Keep torch/torchvision versions matched to avoid "torchvision::nms does not exist".
if [ "$GPU" = "blackwell" ]; then
    pip install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
else
    pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
fi

# 2) lm-polygraph harness (same branch the repo README pins)
pip install "git+https://github.com/IINemo/lm-polygraph.git@supervised_methods"

# 3) extras BEFORE transformers — they pull an older transformers/hub, which we override next.
# datasets==3.6.0 (NOT 5.x): transformers 5.14 pulls datasets 5.x, which dropped
# trust_remote_code / script-dataset support that LM-Polygraph's datasets need; 3.6.0
# matches the team's working env and reads them. Install it last so nothing re-bumps it.
pip install "accelerate>=1.2.0" sentencepiece protobuf safetensors \
    sacrebleu "unbabel-comet" nltk rouge-score sentence-transformers "datasets==3.6.0" hydra-core omegaconf

# 4) pin the DiffusionGemma stack LAST, WITH deps (NOT --no-deps): it must pull the
#    matching huggingface_hub/tokenizers, or transformers 5.14 fails with
#    "cannot import name 'is_offline_mode' from 'huggingface_hub'". This upgrades hub to
#    1.x; unbabel-comet warns it wants transformers<5 / hub<1.0 but still imports — only
#    WMT19's Comet metric depends on it (SamSum/TriviaQA use AlignScore).
#    If `import transformers` raises "torchvision::nms does not exist", the torch/torchvision
#    pair is mismatched — install transformers from source:
#      pip install --force-reinstall git+https://github.com/huggingface/transformers
pip install --upgrade "transformers==5.14.0" "diffusers==0.39.0"
# setuptools<81 provides pkg_resources (comet/torchmetrics import it); importlib_metadata for diffusers.
pip install "setuptools<81" importlib_metadata

# 5) patch unbabel-comet's XLM-R encoder for transformers 5.x. Comet's forward unpacks
#    `last_hidden_states, _, all_layers = self.model(..., return_dict=False)` (a 3-tuple with
#    a pooler output), but transformers 5.x XLM-R returns only 2 positional values, raising
#    `ValueError: not enough values to unpack (expected 3, got 2)` when scoring WMT19/Comet.
#    Switch to return_dict=True and read the named fields — version-robust. Idempotent.
python - <<'PY'
import comet, os
f = os.path.join(os.path.dirname(comet.__file__), "encoders", "xlmr.py")
src = open(f).read()
old = ('        last_hidden_states, _, all_layers = self.model(\n'
       '            input_ids=input_ids,\n'
       '            attention_mask=attention_mask,\n'
       '            output_hidden_states=True,\n'
       '            return_dict=False,\n'
       '        )')
new = ('        _out = self.model(\n'
       '            input_ids=input_ids,\n'
       '            attention_mask=attention_mask,\n'
       '            output_hidden_states=True,\n'
       '            return_dict=True,\n'
       '        )\n'
       '        last_hidden_states = _out.last_hidden_state\n'
       '        all_layers = _out.hidden_states')
if old in src:
    open(f, "w").write(src.replace(old, new))
    print("comet xlmr.py patched for transformers 5.x")
elif "_out.last_hidden_state" in src:
    print("comet xlmr.py already patched")
else:
    print("WARNING: comet xlmr.py forward not in expected shape; patch skipped")
PY

echo "=== pip check (transformers<5.14 warnings from lm-polygraph are EXPECTED; keep the newer transformers) ==="
pip check || true

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
try:
    import diffusers; print("diffusers", diffusers.__version__)
except Exception as e:
    print("diffusers import FAILED:", repr(e)[:120])
# critical: the DiffusionGemma classes must import (else install transformers from source)
try:
    from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor  # noqa: F401
    print("DiffusionGemmaForBlockDiffusion import OK")
except Exception as e:
    print("DiffusionGemma import FAILED (install transformers from source):", repr(e)[:160])
# lm-polygraph must still import atop the newer transformers
try:
    import lm_polygraph  # noqa: F401
    print("lm_polygraph import OK")
except Exception as e:
    print("lm_polygraph import FAILED:", repr(e)[:160])
PY

echo
echo "env ready at: $ENV_PREFIX"
echo "activate with: conda activate $ENV_PREFIX ; export PYTHONNOUSERSITE=1 HF_HOME=$HF_HOME"
