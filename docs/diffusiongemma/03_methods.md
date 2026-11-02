# DiffusionGemma UQ method → stat mapping (design agent 3, source-verified)

All 7 target methods' impls are on branch `feat/dream-minimal` (no other branch needed). Two estimator kinds:
library (bare `name:`, from lm_polygraph@supervised_methods) vs repo-local (`name: methods.<mod>` + `cfg:`).
All trace stats come from `DiffusionGreedyProbsCalculator` (`calculators/diffusion_greedy_probs.py`), whose
`greedy_log_likelihoods` is built from `torch.stack(out.scores)` (line 152) — **this is what breaks (scores=None)**.

| # | Method | Estimator (file:line) | Deps (stat keys) | Sampling/Trace | DG source | Gap/redefine |
|---|---|---|---|---|---|---|
| 1 | MSP | `MaximumSequenceProbability` (lib max_probability.py:8) | greedy_log_likelihoods | single greedy | out.scores | **UNAVAIL** (scores=None). Synthesize greedy_log_likelihoods from committed-token log-probs, OR use TraceProbs. |
| 2 | SemanticEntropy | `SemanticEntropy` (lib semantic_entropy.py:8) | sample_log_probs, sample_texts, semantic_classes_entail | REPEATED SAMPLING (K) | K decodes + NLI | Use `class_probability_estimation="frequency"` → deps drop to [sample_texts, semantic_classes_entail] (avail). |
| 3 | KLE | `KernelLanguageEntropy` (lib kernel_language_entropy.py:53) | semantic_matrix_entail/contra | REPEATED SAMPLING | K decodes → NLI matrix | Trace analog `KernelLanguageEntropyTrace` (methods/kernel_language_entropy_trace.py:53) over generation_trace drafts (1 gen, ≤48² NLI, dedup). |
| 4 | CoCoAMSP | `CocoaMSP` (lib cocoa.py:8) | greedy_sentence_similarity, greedy_log_likelihoods | HYBRID | consistency avail; confidence not | **GAP** no AR likelihood. Repo analog `CocoaMCMaskedNLLTrace(inf_method=TraceProbs/ConfidenceAtCommit)`. Formula (-Σ gll)·mean(1-sim). |
| 5 | TraceEntropy | `TraceEntropy` (methods/trace_entropy.py:7) | gen_trace_entropy | TRACE | **DIRECT 1:1** out.gen_trace_entropy | none, ~free. |
| 6 | ConfidenceAtCommit | `ConfidenceAtCommitment` (methods/confidence_at_commitment.py:7) | confidence_at_commit | TRACE, ~free | out.confidence_at_commit | **REDEFINE**: commit=argmax-stabilization (not first-unmask, llada/generate.py:258-267). Same stat key, new commit def. Feeds #6 and #7. |
| 7 | D-CoCoA-L = `CocoaConfidenceAtCommitCrossEncoderBlockAvgTrace` | `CocoaMCMaskedNLLTrace` (methods/cocoa_mc_masked_nll_trace.py:9; cfg inf_method=ConfidenceAtCommit, similarity_metric=cross_encoder, block_only=true, block_agg=avg) | greedy_sentence_similarity_block, confidence_at_commit, greedy_texts | TRACE | per-block cross-enc over block_generation_traces + confidence_at_commit | inherits #6 redefinition; cross-enc over deduped block drafts. |

## Implementation order (cheapest first)
1. TraceEntropy, TraceProbs, ConfidenceAtCommit — ~free; need trace-collector + the commit-step redefinition.
2. D-CoCoA-L + Cocoa-trace — cross-enc over deduped block_generation_traces; depends on #1's redefined confidence_at_commit.
3. KLETrace / LUQTrace — NLI over deduped drafts (SemanticMatrixTraceCalculator ← generation_trace).
4. Sampling-text calculator → SemanticEntropy(frequency), KLE, LUQ, CoCoAMSP consistency — ~Kx gens + K² NLI.
5. MSP, CoCoAMSP confidence, SemanticEntropy sample_log_probs — via synthesized greedy_log_likelihoods.

## Two root gaps (everything reduces to these)
- **scores=None** → greedy_log_likelihoods absent → MSP, CoCoAMSP-confidence, SemanticEntropy sample_log_probs.
  FIX: backend synthesizes greedy_log_likelihoods = per-committed-token log-prob (= confidence_at_commit per pos) —
  the diffusion-native sequence likelihood. Unlocks MSP/Perplexity/CoCoAMSP through UNCHANGED estimator code.
- **No persistent mask / recomputed acceptance** → confidence_at_commit REDEFINED via argmax-stabilization.
  One key, feeds ConfidenceAtCommit(#6) + D-CoCoA-L(#7).
Trace-native methods (TraceEntropy, KLETrace, LUQTrace, Cocoa-trace, ConfidenceAtCommit) map cleanly; only added
compute is cross-encoder/NLI over DEDUPED drafts.
