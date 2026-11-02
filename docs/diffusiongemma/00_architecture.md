# DiffusionGemma UQ — Architecture (synthesis of 4 design agents)

**Why:** NeurIPS rebuttal, reviewer weakness (3) — show the paper's UQ methods generalize to a **non-masked,
block/canvas diffusion LM** (`google/diffusiongemma-26B-A4B-it`) beyond masked-diffusion LLaDA/Dream.
Methods: MSP, SemanticEntropy, KLE, CoCoAMSP, TraceEntropy, ConfidenceAtCommit, D-CoCoA-L. Datasets: SamSum, WMT19, TriviaQA.

## The one enabling insight
The harness is **stat-key-driven**: estimators/trace-calculators consume named stats and never touch the model.
So the entire job = **produce the standard stat keys from DiffusionGemma's denoising trace**; then KLE, LUQ,
D-CoCoA-L, TraceEntropy, ConfidenceAtCommit run **unchanged** (agent 1). Only NEW code: a backend, a loader, an
aggregate-only calculator, 3 configs, 1 dispatch branch (agents 1+4).

## Two load-bearing design decisions (resolve every gap — agent 3)
1. **Synthesize `greedy_log_likelihoods` from committed-token log-probs.** DiffusionGemma `generate()` returns
   `scores=None` (agent 2), so there is no AR sequence likelihood. Define the diffusion-native likelihood as the
   per-position log-prob of each committed token **at its commit step**. This single synthesized stat unlocks MSP,
   Perplexity, and CoCoAMSP's confidence term through the **unchanged** library estimators.
2. **`confidence_at_commit` via argmax-stabilization.** No persistent mask; acceptance is non-monotonic (agent 2).
   Redefine commit: `commit_step(p) = 1 + max{s : argmax_s[p] != final[p]}` (0 if always matches);
   `confidence_at_commit[p] = log P(argmax = final[p]) at commit_step`. Same stat key; new collector logic.
   Feeds ConfidenceAtCommit (#6) and D-CoCoA-L (#7).

## Components (new files; branch `feat/diffusiongemma` off `main`)
| File | Role |
|---|---|
| `llada/load_diffusiongemma.py` | `load_model` → `DiffusionGemmaForBlockDiffusion.from_pretrained(bf16, device_map)`, `load_tokenizer` → `AutoProcessor`; register EOS `[1,106,50]`. |
| `whitebox_diffusiongemma.py` | `WhiteboxModelDiffusionGemma(Model)` — duck-types `WhiteboxModelDiffusionLLM` (agent 1 §1): `.model_type="CausalLM"`, `tokenize/generate/generate_texts/device/__call__`. `generate()` runs the OFFICIAL block-diffusion loop with a read-only `UQTraceProcessor` (registered first) + `put_draft` streamer (agent 2), then assembles the output object with the 14 diffusion attrs + synthesized `greedy_log_likelihoods` + a `scores`-like tuple from commit-step reductions. **Does NOT import `llada.generate`.** |
| `calculators/diffusiongemma_greedy_probs.py` (+ `_builder.py`) | aggregate-only calculator: emits the stat keys (agent 1 §2) from the backend output; `store_full_step_logits=False`; DROP full `greedy_log_probs` (Gemma 256k vocab = 131 MB/sample, agent 4); never `torch.stack(out.scores)`. |
| `examples/configs/model/diffusiongemma.yaml` | `type: DiffusionGemma`, `path_to_load_script`, `device_map: auto`, `eos_ids:[1,106,50]`. |
| `examples/configs/estimators/estimators_diffusiongemma.yaml` | the 7 methods (agent 3 forms). |
| `examples/configs/stat_calculators/calculators_diffusiongemma.yaml` | clone `new_calculators_llada_no_train.yaml`; drop MC/Sample/Block MaskedLL; swap in DiffusionGemma calc; keep SemanticMatrixTrace + CrossEncoder/Deberta trace calcs. |
| `polygraph_eval` | add `elif model.type=="DiffusionGemma": get_diffusion_gemma_model(args)` (~line 373) + the constructor (~524). |
| `scripts/setup_diffusiongemma_env.sh`, `scripts/slurm/submit_diffusiongemma.sh`, `scripts/validate_diffusiongemma.py` | isolated env, Slurm 2×A100, runtime smoke/verification. |

## Backend output-object contract (agent 1 §1/§2)
`generate()` returns object with: `.sequences (B, prompt+gen)`; `.scores` tuple (synthesized, log-softmaxed) or a
guard so the DG calculator doesn't need it; and diffusion attrs `.nfe .remask_count .first_remask_step
.generation_trace .full_generation_trace .last_block_generation_trace .block_generation_traces .gen_trace_logits
.gen_trace_entropy .gen_trace_argmax .gen_trace_top2gap .confidence_at_commit .mask_fraction_at_commit .mask_id`.
Trace tensors packed `(B, num_canvases, max_steps, canvas_length)`; drafts `List[batch][List[str]]` (deduped).

## Method → mechanism (agent 3)
Trace-native, ~free: **TraceEntropy** (gen_trace_entropy), **TraceProbs** (gen_trace_logits), **ConfidenceAtCommit**
(redef #2). Trace + NLI/cross-enc over deduped drafts (1 gen): **KLE-Trace**, **LUQ-Trace**, **D-CoCoA-L**
(CocoaMCMaskedNLLTrace, inf=ConfidenceAtCommit, sim=cross_encoder, block_only, avg). Sampling (K≈10 gens):
**SemanticEntropy** (use `frequency` variant → no sample_log_probs), **KLE**, **LUQ**, **CoCoAMSP** consistency.
Likelihood-based via synthesized gll (#1): **MSP**, **CoCoAMSP** confidence.

## Env / GPU / storage (agent 4)
Isolated conda-prefix env: torch(cu124/cu128) → lm-polygraph@supervised_methods → force `transformers==5.14`
(floor 5.11) + `diffusers==0.39` `--no-deps` → `pip check`. GPU: RunPod-97GB single-GPU (iterate) or 2×A100-40GB
device_map (sweep); never materialize step logits (64 GB OOM). Store aggregates only; drop full greedy_log_probs.
Rollout: smoke(subsample=8) → trace-calc validation → cheap methods → sampling/NLI → 3-dataset sweep (~24-36 A100h).

## RUNTIME-VERIFY before trusting (agent 2 §5) — can't be checked without the model/env
Exact `logits_processor(input_ids, scores, cur_step=)` signature, scores `(B,256,V)`, once/step, user-runs-first;
`put_draft(value=ids vs logits=)`; identity-return byte-identical vs no-processor; `out.scores is None` +
`tokens_per_forward`; re-read `generation_config.json` from the loaded model; `EntropyBoundSampler` importable;
`diffusers==0.39.0` pin. The code marks these `# VERIFY@runtime`.
