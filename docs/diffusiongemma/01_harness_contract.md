# DiffusionGemma → UQ Harness Integration Contract (design agent 1)

All paths under repo root (branch `feat/dream-minimal`). Cited `file:line`.

## 1. Backend interface — `DiffusionGemmaWhiteboxModel`
Reference: `WhiteboxModelDiffusionLLM(Model)` in `whitebox_model_dllm.py:19` (subclasses `lm_polygraph.utils.model.Model`). Duck-type:

**Constructor attrs** (`whitebox_model_dllm.py:22-92`): `.model`, `.tokenizer` (pad forced), `.generation_parameters`, `.stop_sequences`/`.stop_token_ids`, `.steps`, `.length`, `.block_length`, `.mask_id`, `.use_cache`, `.base_device`, `.model_type="CausalLM"` (**hardcoded, load-bearing**), `.instruct`, `.attention_device`, `.needs_attention_mask`, `.shift_logits`, `.prompt_suffix`.

**Methods**: `device()`→base_device (184); `tokenize(texts)`→dict w/ input_ids, chat-template when instruct, left-pad (187-224); `generate(**kw)`→output object (94-182); `generate_texts()`→List[str] decoding `.sequences` (232-238); `__call__(**args)`→`self.model(**args)` (226-230); `post_processing(...)` builds output (335-523).

**generate() kwargs** (from `calculators/diffusion_greedy_probs.py:133-151`): input_ids, min_new_tokens, max_new_tokens, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, num_return_sequences, suppress_tokens, do_sample; reads generation_parameters.temperature.

**Return object** (`GenerateDecoderOnlyOutput`, 459-516): `.sequences` `(B, prompt+gen)`; `.scores`/`.logits` tuple len n_tokens each `(B,vocab)` **already log_softmax+fp16**; `.attentions`, `.hidden_states`; then diffusion attrs: `.nfe`, `.remask_count`, `.first_remask_step`, `.generation_trace`, `.full_generation_trace`, `.last_block_generation_trace`, `.block_generation_traces`, `.gen_trace_logits`, `.gen_trace_entropy`, `.gen_trace_argmax`, `.gen_trace_top2gap`, `.confidence_at_commit`, `.mask_fraction_at_commit`, `.mask_id`. These come from the backend gen fn's 3-tuple `(x, nfe, remask_stats)` (`llada/generate.py:405`; remask_stats keys 391-404). **DiffusionGemma sampler must emit this 3-tuple or rewrite post_processing.**

## 2. Stat-key contract — reuse `DiffusionGreedyProbsCalculator` (`calculators/diffusion_greedy_probs.py`)
Always (215-226): input_tokens, greedy_log_probs `List[np(seq,vocab)]`, greedy_tokens, greedy_tokens_alternatives (n=10), greedy_texts, greedy_log_likelihoods, attention_raw+tokenizer, embeddings_decoder_raw. Slicing at 179-185 keys off `model_type=="CausalLM"`.
Diffusion (if `output_diffusion_stats=True`, 228-243): the 14 attrs pulled verbatim off the output object.
Shapes: gen_trace_{logits,entropy,argmax,top2gap} `(B,num_blocks,max_steps,block_length)`; confidence_at_commit/mask_fraction_at_commit `(B,gen_len)`; remask_count `(B,prompt+gen)`; first_remask_step `(B,gen_len)`; generation_trace/full/last_block `List[batch][List[draft_str]]`; block_generation_traces `List[batch][block][step_str]`.
Builder `calculators/diffusion_greedy_probs_builder.py:21` reusable if attrs match.

## 3. Wiring points
1. Dispatch `polygraph_eval:362-380` (`get_model`); DiffusionLLM→get_diffusion_model at 373. **Option B** (needed, DG not drivable by llada.generate): add `elif=="DiffusionGemma": get_diffusiongemma_model` + a `get_diffusiongemma_model` modeled on `get_diffusion_model` (524-550).
2. New `llada/load_diffusiongemma.py` w/ `load_model`/`load_tokenizer` (pattern `llada/load_dream.py:5,22`).
3. Config keys (524-549): path_to_load_script, path, load_model_args, generation_params, load_tokenizer_args, device, attention_device, mask_id(def 126336), use_cache(def True), prompt_suffix, shift_logits(def False), instruct.
4. `get_stat_calculator_names` 229-235 buckets any non-Blackbox type into "Whitebox" — new model.type works w/o edit.
5. Calculator YAML entries: name, builder module path, cfg, stats, dependencies.
6. CLI ref `run_experiments.sh:41`.

Plugin: builder exposes `load_stat_calculator(config, builder_env)`→StatCalculator; calculator has `meta_info()→(stats,deps)` and `__call__(dependencies: dict,...)`. Estimators: YAML `name: methods.<mod>`; FactoryEstimator imports + calls `load_estimator(cfg)`; deps via `super().__init__([keys],"sequence")`, read `stats[key]`.

## 4. Trace-stat contract (D-CoCoA-L, KLE, LUQ, TraceEntropy, ConfidenceAtCommit)
Backend supplies drafts (generation_trace/full/last_block/block as `List[batch][List[str]]`) + per-step tensors (gen_trace_logits/entropy `(B,nb,steps,blk)`) + confidence_at_commit `(B,gen)`. Then these run UNCHANGED:
- Trace calcs: SemanticMatrixTraceCalculator (reads generation_trace → semantic_matrix_{entail,contra,classes,*_logits}_trace, entailment_id); GreedyCrossEncoderSimilarityMatrix*Trace (→ greedy_sentence_similarity_*_trace); GreedyDeberta*Trace. NLI=microsoft/deberta-large-mnli; cross-enc=cross-encoder/stsb-roberta-large. Matrices `(B,n,n)`, n=#unique drafts.
- Estimators: KLE=`KernelLanguageEntropyTrace`(semantic_matrix_entail/contra_trace); LUQ=`LUQTrace`(semantic_matrix_entail/contra_logits_trace); D-CoCoA=`CocoaMCMaskedNLLTrace`(greedy_sentence_similarity_* or greedy_deberta_* + inf_stat + greedy_texts + nfe); ConfidenceAtCommit=`ConfidenceAtCommitment`(confidence_at_commit); TraceEntropy=`TraceEntropy`(gen_trace_entropy).

## Minimal drop-in checklist
1. Backend duck-typing §1 (model_type="CausalLM", the 14 diffusion attrs, log-softmaxed scores).
2. `llada/load_diffusiongemma.py`.
3. Reuse DiffusionGreedyProbsCalculator+builder if attrs match, else DiffusionGemma variant emitting identical keys/shapes.
4. Dispatch branch at polygraph_eval:373 + get_diffusiongemma_model at ~524.
