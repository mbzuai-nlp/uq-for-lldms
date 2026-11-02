# DiffusionGemma env / config / infra plan (design agent 4)

## ENV (isolated prefix env, NOT the working `uncertainty_dllm` env)
```bash
export DG_ROOT=/lustre/$USER/diffgemma; export ENV_PREFIX=$DG_ROOT/envs/diffgemma
export HF_HOME=$DG_ROOT/hf_cache; mkdir -p "$DG_ROOT/envs" "$HF_HOME"
conda create -y -p "$ENV_PREFIX" python=3.11 && conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124  # A100; Blackwell→cu128 torch2.7
pip install "git+https://github.com/IINemo/lm-polygraph.git@supervised_methods"
pip install --upgrade --force-reinstall --no-deps "transformers==5.14.0" "diffusers==0.39.0"
pip install "accelerate>=1.2.0" sentencepiece protobuf safetensors sacrebleu unbabel-comet nltk \
    rouge-score sentence-transformers datasets hydra-core omegaconf
pip check   # lm-polygraph may pin transformers<5.14 → keep newer, smoke-import lm_polygraph.estimators
```
Verified floor is transformers **5.11** (agent 2); if `torchvision::nms` import error → install transformers from source.

## New files (5) + 3 code hooks
- `examples/configs/model/diffusiongemma.yaml`: `type: DiffusionGemma`, `path: google/diffusiongemma-26B-A4B-it`,
  `path_to_load_script: llada/load_diffusiongemma.py`, `load_model_args:{device_map: auto}`, `eos_ids:[1,106,50]`.
- `examples/configs/estimators/estimators_diffusiongemma.yaml`: the 7 methods (trace_entropy avg/max, trace_probs
  avg/max, LUQ, KernelLanguageEntropy, luq_trace, kernel_language_entropy_trace, confidence_at_commitment avg/min).
- `examples/configs/stat_calculators/calculators_diffusiongemma.yaml`: clone new_calculators_llada_no_train.yaml;
  DROP MC/Sample/Block MaskedLL calculators; replace DiffusionGreedyProbs → `DiffusionGemmaGreedyProbsCalculator`
  (builder `calculators.diffusiongemma_greedy_probs_builder`, `store_full_step_logits:False`, DROP full
  `greedy_log_probs`); KEEP SemanticMatrixTrace + GreedyCrossEncoder*/GreedyDeberta* trace calcs (power LUQ/KLE-trace).
- `llada/load_diffusiongemma.py`: `AutoProcessor.from_pretrained` + `DiffusionGemmaForBlockDiffusion.from_pretrained
  (torch_dtype=bf16, device_map=device_map)`; set pad; register EOS [1,106,50]; return (model, processor).
- 3 code hooks: `polygraph_eval get_model` add `elif type=="DiffusionGemma": get_diffusion_gemma_model`;
  new `WhiteboxModelDiffusionGemma(Model)` (does NOT import llada.generate; runs official block-diffusion loop with
  the UQTraceProcessor/streamer; returns output object w/ the 14 diffusion attrs); `calculators/diffusiongemma_greedy_probs.py(+_builder)`.

## CLI (reuse the 3 dataset configs unchanged; `model=diffusiongemma` overrides the default bloomz)
```
HYDRA_CONFIG=.../polygraph_eval_samsum_simple_instruct.yaml python polygraph_eval batch_size=1 \
  subsample_eval_dataset=100 model=diffusiongemma estimators=estimators_diffusiongemma \
  stat_calculators=calculators_diffusiongemma cache_path=./workdir/dg_samsum
# also wmt19_deen_simple_instruct (nmt→Comet), triviaqa_simple_instruct (qa→AlignScore)
# NLI tight? append +attention_device=cpu +deberta_batch_size=1
```

## GPU (51.7 GB bf16; docs say >60 GB single-GPU)
- Single 47 GB Ada: NO. RunPod 97 GB Blackwell: single-GPU, BEST for iteration (needs cu128 torch).
  2×A100 40 GB Slurm: device_map=auto shard (~25.9 GB/GPU), BEST for sweep. GB10 unified ~128 GB fallback.
- Never materialize full canvas logits (256 MiB/step × 256 ≈ 64 GB OOM). NLI DeBERTa ~1.6 GB + cross-enc ~1.4 GB
  on GPU0 headroom or offload to CPU. `device_map=auto`, fallback explicit max_memory (llada/load_llada.py pattern).

## Trace storage (bound RAM + pickle)
Keep aggregates only: gen_trace_{entropy,argmax,top2gap}, confidence_at_commit, mask_fraction_at_commit, nfe,
remask_count, first_remask_step, deduped drafts. NEVER keep gen_trace_logits. DROP full greedy_log_probs
(Gemma 256k vocab = 131 MB/sample) — keep only greedy_tokens_alternatives top-k. ~30 MB/sample →
100-sample smoke ≈ 3-5 GB. Keep subsample 200-500; cache_path on Lustre not $HOME; one pickle per (dataset,seed).

## Slurm 2×A100 template → scripts/slurm/submit_diffusiongemma.sh (adapt submit_dream.sh)
`#SBATCH --gres=gpu:a100:2 --mem=192G -p long`; conda activate the prefix env; PYTHONNOUSERSITE=1;
HF_HOME=Lustre; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; run polygraph_eval as above.
RunPod: `setsid nohup ... > log 2>&1 &` detached.

## Rollout: (1) gen-parity smoke subsample=8 RunPod → (2) trace-calc validation → (3) cheap methods
(trace_entropy/probs, confidence_at_commit) → (4) sampling/NLI (LUQ/KLE ~10× gen; LUQ/KLE-trace 1 gen but
~48² NLI/canvas, dedup) → (5) full 3-dataset sweep subsample 200-500, ~24-36 A100-hours.

## Risks: env pin conflict (force-reinstall+pip check); Blackwell needs cu128; sharding OOM (explicit max_memory);
sampling+NLI co-resident OOM (cpu offload, dedup); pickle blowup (drop greedy_log_probs); scores=None crash
(new calculator must not torch.stack(out.scores)); new model.type needs the dispatch+class+builder before any run.

## Metric compat: WMT19's Comet (unbabel-comet) breaks under transformers 5.x — its XLM-R encoder
(`comet/encoders/xlmr.py` forward) unpacks `last_hidden_states, _, all_layers = self.model(..., return_dict=False)`
(a 3-tuple with pooler output), but transformers 5.x XLM-R returns only 2 positional values →
`ValueError: not enough values to unpack (expected 3, got 2)`. With batch_size=1 the harness scores Comet
per-sample, so it aborts on the first sample (pickle ends up n=1; generation is NOT at fault). Fix: switch that
forward to `return_dict=True` and read `.last_hidden_state` / `.hidden_states`. Applied reproducibly in
`scripts/setup_diffusiongemma_env.sh` (step 5, idempotent). SamSum/TriviaQA use AlignScore and are unaffected.
