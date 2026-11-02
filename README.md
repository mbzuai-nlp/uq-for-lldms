# Uncertainty Quantification for Large Language Diffusion Models

**NeurIPS 2026** · [Paper](https://arxiv.org/abs/2605.14570) · [Project page](https://mbzuai-nlp.github.io/uq-for-lldms/) · [BibTeX](#citation)

Artem Vazhentsev, Vladislav Smirnov, David Li, Maxim Panov, Timothy Baldwin, Artem Shelmanov
(MBZUAI, The University of Melbourne)

Large language diffusion models (LLDMs) generate text by iterative denoising rather than left to right, so
uncertainty quantification (UQ) methods designed for autoregressive LLMs either do not apply or require costly
repeated sampling. This repository contains the benchmark of UQ methods for LLDMs, the diffusion-specific
uncertainty signals (masked-diffusion likelihood, commitment confidence, trajectory statistics, trajectory
semantic instability), and **D-CoCoA** (D-CoCoA-G and D-CoCoA-L), which replaces repeated sampling with the
intermediate states of a single denoising trajectory. Everything is built on top of
[LM-Polygraph](https://github.com/IINemo/lm-polygraph).

Supported models: [LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5),
[Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B),
[ReFusion](https://huggingface.co/GSAI-ML/ReFusion) and DiffusionGemma-26B-A4B (uniform-state diffusion, no mask token).

## Installation

Install `lm-polygraph` from the custom branch:

```bash
pip install git+https://github.com/IINemo/lm-polygraph.git@supervised_methods
```

The code has been tested with Python 3.11, PyTorch 2.11.0 and Transformers 5.8.1.
DiffusionGemma needs a newer stack (Transformers 5.14, Diffusers 0.39), see
[`docs/diffusiongemma/04_infra.md`](docs/diffusiongemma/04_infra.md).

## Repository structure

| Path | Contents |
|---|---|
| [`whitebox_model_dllm.py`](whitebox_model_dllm.py) | LM-Polygraph adapter for LLDMs |
| [`whitebox_diffusiongemma.py`](whitebox_diffusiongemma.py) | adapter for DiffusionGemma |
| [`llada/generate.py`](llada/generate.py), [`refusion/generate.py`](refusion/generate.py) | generation with diffusion statistics (adapted from [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) and [ReFusion](https://github.com/ML-GSAI/ReFusion)) |
| [`calculators/`](calculators) | statistic calculators (masked-diffusion likelihood, trajectory similarities, ...) |
| [`methods/`](methods) | diffusion-specific UQ methods, including D-CoCoA |
| [`examples/configs/estimators/all_estimators_llada_complete.yaml`](examples/configs/estimators/all_estimators_llada_complete.yaml) | full list of evaluated methods (~600 variants) |
| [`examples/configs/`](examples/configs) | dataset configs; for instruction-tuned models use `*_simple_instruct.yaml` |
| [`docs/METHODS.md`](docs/METHODS.md) | formulas of the UQ methods |
| [`notebooks/`](notebooks) | result tables and figures |

## Running experiments

Evaluate LLaDA-1.5 on GSM8K with all methods:

```bash
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_gsm8k_simple_instruct.yaml python polygraph_eval batch_size=1 subsample_eval_dataset=2000 model.path=GSAI-ML/LLaDA-1.5 model.path_to_load_script=`pwd`/llada/load_llada.py estimators=all_estimators_llada_complete stat_calculators=new_calculators_llada_new_baselines model.type=DiffusionLLM +model.device=cuda cache_path=./workdir/output +log_time=True +attention_device=cpu stat_calculators.14.cfg.subsample_train_dataset=500 stat_calculators.14.cfg.subsample_background_train_dataset=500 stat_calculators.14.cfg.target_metric=Accuracy
```

Dream:

```bash
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_gsm8k_simple_instruct.yaml python polygraph_eval batch_size=1 subsample_eval_dataset=2000 model.path=Dream-org/Dream-v0-Instruct-7B model.path_to_load_script=`pwd`/dream/load_dream_cached.py estimators=all_estimators_llada_complete stat_calculators=new_calculators_llada_new_baselines model.type=DiffusionLLM +model.device=cuda cache_path=./workdir/output +log_time=True +attention_device=cpu +model.shift_logits=true +model.use_cache=true +model.mask_id=151666 stat_calculators.14.cfg.subsample_train_dataset=500 stat_calculators.14.cfg.subsample_background_train_dataset=500 stat_calculators.14.cfg.target_metric=Accuracy
```

ReFusion:

```bash
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_gsm8k_simple_instruct.yaml python polygraph_eval batch_size=1 model=refusion model.path=GSAI-ML/ReFusion model.path_to_load_script=`pwd`/refusion/load_refusion.py model.type=DiffusionLLM +model.device=cuda model.mask_id=151670 model.use_cache=true subsample_eval_dataset=2000 estimators=all_estimators_llada_complete stat_calculators=new_calculators_llada_new_baselines cache_path=./workdir/output stat_calculators.14.cfg.subsample_train_dataset=500 stat_calculators.14.cfg.subsample_background_train_dataset=500 stat_calculators.14.cfg.target_metric=Accuracy +attention_device=cpu +log_time=True ++trust_remote_code=false
```

DiffusionGemma (methods that need a mask token are excluded from its configs):

```bash
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_gsm8k_simple_instruct.yaml python polygraph_eval batch_size=1 subsample_eval_dataset=2000 model=diffusiongemma model.path_to_load_script=`pwd`/llada/load_diffusiongemma.py ignore_exceptions=true estimators=estimators_diffusiongemma_complete stat_calculators=calculators_diffusiongemma_complete cache_path=./workdir/output +log_time=True +attention_device=cpu stat_calculators.14.cfg.subsample_train_dataset=500 stat_calculators.14.cfg.subsample_background_train_dataset=500 stat_calculators.14.cfg.target_metric=Accuracy
```

[`scripts/run_local/run_uq.sh`](scripts/run_local/run_uq.sh) wraps these commands with the per-dataset settings used
in the paper (training-subsample sizes, target metrics, thresholds for COMET):

```bash
scripts/run_local/run_uq.sh llada gsm8k 0
```

## Results

Result tables and the performance-vs-runtime figures are built in
[`notebooks/results_clean.ipynb`](notebooks/results_clean.ipynb) (LLaDA-1.5, Dream) and
[`notebooks/results_refusion.ipynb`](notebooks/results_refusion.ipynb) (ReFusion).

## Citation

```bibtex
@article{vazhentsev2026uncertainty,
  title   = {Uncertainty Quantification for Large Language Diffusion Models},
  author  = {Vazhentsev, Artem and Smirnov, Vladislav and Li, David and
             Panov, Maxim and Baldwin, Timothy and Shelmanov, Artem},
  journal = {arXiv preprint arXiv:2605.14570},
  year    = {2026}
}
```
