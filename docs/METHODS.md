# Uncertainty Quantification Methods for Diffusion LLMs

## Overview

All methods estimate **sequence-level** uncertainty. Higher score = more uncertain. Methods are grouped by what data they need and whether they cost extra compute beyond generation.

Total: **55+ methods** (including variants) across 8 categories.

## Available Trace Data (collected during generation for free)

| Trace | Shape | Description |
|-------|-------|-------------|
| `gen_trace_logits` | `(B, blocks, steps, block_len)` | Max log-prob per token at each diffusion step |
| `gen_trace_entropy` | same | Entropy of probability distribution at each step |
| `gen_trace_argmax` | same, dtype long | Argmax token ID at each step |
| `gen_trace_top2gap` | same | Gap between top-2 token probabilities at each step |
| `confidence_at_commit` | `(B, gen_len)` | Log-prob at the exact step each token was committed |
| `mask_fraction_at_commit` | `(B, gen_len)` | Fraction of sequence still masked when each token was committed |
| `remask_count` | `(B, L)` | Times each token was remasked during generation |
| `first_remask_step` | `(B, L)` | Step when each token was first committed |
| `nfe` | scalar | Number of function evaluations (forward passes) |
| `generation_trace` | list of strings | Text output at each step |

---

## Category 1: Diffusion Trace Methods (free, no extra forward passes)

These methods extract uncertainty signals directly from the diffusion generation process.

### 1a. Token Confidence Across Steps

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **TraceProbs** | `gen_trace_logits` | Average/max of per-step max log-probs across all steps. Basic confidence measure. | avg, max |
| **TraceEntropy** | `gen_trace_entropy` | Average/max of per-step entropy across all steps. High entropy = flat distribution = uncertain. | avg, max |
| **LastStepConfidence** | `gen_trace_logits` | Max log-prob at the **final step of the block** (fewest tokens remain masked). | avg, min |
| **Top2GapTrace** | `gen_trace_top2gap` | Gap between top-2 token probabilities. Small gap = model torn between two candidates. | avg, min |

### 1b. Confidence Dynamics Across Steps

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **TraceNLLVariance** | `gen_trace_logits` | **Variance** of max log-prob across steps per token. High variance = oscillating confidence. | avg, max |
| **ConfidenceSlope** | `gen_trace_logits` | Linear trend (last - first) / steps. Negative slope = confidence declining with more context. | avg, min |
| **ConfidenceCurvature** | `gen_trace_logits` | **Second derivative** of log-probs across steps. Captures backtracks and over-corrections that first-order slope misses. Inspired by score curvature in continuous diffusion theory. | avg, max |
| **ArgmaxFlipCount** | `gen_trace_argmax` | Counts how many times the predicted token **identity changes** across steps. More flips = more uncertain. | avg, max, sum |

### 1c. Commitment-Point Methods (novel, unique to dLLMs)

These capture information at each token's individual decision point — the exact moment it is committed and no longer remasked. **No autoregressive equivalent exists.**

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **ConfidenceAtCommitment** | `confidence_at_commit` | Log-prob at the exact step each token is committed. Unlike LastStepConfidence (block-level last step), this is per-token. | avg, min |
| **MaskNeighborRatio** | `mask_fraction_at_commit` | Fraction of the sequence still masked when each token is committed. Tokens committed with little revealed context are more uncertain. | avg, max |
| **CommitmentSynchrony** | `first_remask_step` | How synchronized token commits are. Spike (many tokens at once) = collective decision = certain. Spread = dithering. Sequence-level structural signal. | entropy, peak_ratio, concentration |

### 1d. Convergence & Stability Analysis (novel, theoretically grounded)

These apply established theoretical frameworks (phase transitions, MCMC convergence, rank stability) to the dLLM generation trajectory. **No prior application to diffusion LLM generation exists.**

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **PhaseTransitionSharpness** (jump) | `gen_trace_logits` + `gen_trace_argmax` | Magnitude of the confidence jump at each token's stabilization step. Large abrupt jump = forced commitment. | avg, max |
| **PhaseTransitionSharpness** (width) | same | Number of steps for confidence to go from 20% to 80% of final value. Wider = more gradual = more uncertain. | avg, max |
| **PhaseTransitionSharpness** (late_stability) | same | Fraction of total steps after which the token stabilizes. Later = more uncertain. | avg, max |
| **TokenRankStability** (consistency) | `gen_trace_argmax` | Fraction of steps where the argmax matches the final committed token. Low = the model changed its mind. | avg, min |
| **TokenRankStability** (first_appearance) | `gen_trace_argmax` | Step at which the committed token first becomes the argmax (fraction). Late first appearance = surprise winner. | avg, min |
| **MCMCDiagnostics** (split_rhat) | `gen_trace_logits` | Split-half R-hat (Gelman-Rubin). Compares first-half vs second-half variance. R-hat >> 1 = chain hasn't converged. | avg, max |
| **MCMCDiagnostics** (ess) | `gen_trace_logits` | Effective sample size normalized by chain length. Low ESS = highly autocorrelated = model is stuck, not genuinely confident. | avg, max |

### 1e. Remasking Statistics

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **RemaskCountStat** | `remask_count` | How many times each token was remasked. More remaskings = harder token. | mean, max, sum |
| **FirstRemaskStepCount** | `first_remask_step` | Counts tokens never committed (-1 values), or mean step of first commitment. | count, mean_step |
| **NFEUncertainty** | `nfe` | Total forward passes used during generation. More steps needed = harder example. | (scalar) |

### 1f. Multi-Scale & Positional Methods

These capture structural patterns in how uncertainty is distributed across the generated sequence.

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **BlockUncertaintyVariance** (logprob) | `gen_trace_logits` | Variance of mean log-prob **across blocks**. High variance = some blocks much harder than others. Leverages semi-AR block structure. | var, range |
| **BlockUncertaintyVariance** (entropy) | `gen_trace_entropy` | Same as above using entropy. | var, range |
| **PositionConfidenceCorrelation** | `confidence_at_commit` | Spearman/Pearson correlation between token position and confidence. Detects systematic positional bias (e.g., later tokens always less confident). Inspired by AlphaFold pLDDT. | spearman, pearson |

---

## Category 2: Lexical Similarity Methods (free)

Compare text outputs across diffusion steps using lexical overlap metrics.

| Method | Stat | What it measures | Aggregation |
|--------|------|-----------------|-------------|
| **LexicalSimilarityTrace** | `generation_trace` | Negative mean pairwise ROUGE/BLEU between intermediate generations. Low similarity between steps = unstable output. | rouge1, rouge2, rougeL, BLEU |

---

## Category 3: Log-Likelihood Methods (require MC masked evaluation)

Probe the model by randomly masking tokens and measuring how well it can reconstruct them.

| Method | Stat | What it measures | Extra cost |
|--------|------|-----------------|------------|
| **MCMaskedNLL** | `mc_masked_ll` | Negative log-likelihood under random mask perturbations. High NLL = model struggles to reconstruct = uncertain. | ~10 forward passes |
| **MCMaskedNormalizedNLL** | `mc_masked_normalized_ll` | Same as above, normalized per token. | ~10 forward passes |
| **BlockMCMaskedNLL** | `block_mc_masked_ll` | Block-wise masked NLL for semi-autoregressive models. | ~10 forward passes |

---

## Category 4: Semantic Similarity Methods (require NLI model)

Build a semantic similarity graph between intermediate generations and analyze its structure.

| Method | Stat | What it measures | Extra cost |
|--------|------|-----------------|------------|
| **DegMatTrace** | semantic matrices | Trace of degree matrix (sum of pairwise similarities). Low degree = isolated = uncertain. | NLI model inference |
| **EigValLaplacianTrace** | semantic matrices | Sum of eigenvalues of graph Laplacian. Captures graph connectivity. | NLI model inference |
| **EccentricityTrace** | semantic matrices | Frobenius norm of informative eigenvectors. Detects outlier generations. | NLI model inference |
| **NumSemSetsTrace** | semantic matrices | Number of connected components in similarity graph. More clusters = more ambiguity. | NLI model inference |
| **LUQTrace** | semantic NLI logits | Mean pairwise semantic similarity from NLI logits. | NLI model inference |
| **KernelLanguageEntropyTrace** | semantic matrices | Von Neumann entropy of heat kernel of semantic graph. | NLI model inference |
| **AvgDissimilarityTrace** | cross-encoder/DeBERTa scores | Average dissimilarity between generations using cross-encoder or DeBERTa NLI. Supports progress weighting. | NLI/cross-encoder inference |

---

## Category 5: Composite Methods (combine likelihood + similarity)

Multiply or combine a likelihood signal with a semantic/lexical signal.

| Method | Components | What it measures |
|--------|-----------|-----------------|
| **CocoaMCMaskedNLL** | MCMaskedNLL + cross-encoder similarity | NLL weighted by semantic dissimilarity. High NLL + high dissimilarity = very uncertain. |
| **CocoaMCMaskedNormalizedNLL** | MCMaskedNormalizedNLL + similarity | Normalized version of above. |
| **CocoaMCMaskedNLLTrace** | TraceProbs/TraceEntropy + similarity trace | Combines trace-based inference with trace-based similarity. Supports progress weighting and NFE multiplication. Most configurable method. |

---

## Category 6: Sample-Based Methods (require multiple generations)

Generate multiple outputs via sampling, then analyze agreement.

| Method | Stat | What it measures | Extra cost |
|--------|------|-----------------|------------|
| **SemanticEntropyMCMaskedNLL** | sample NLLs + semantic classes | Entropy over semantic equivalence classes. Groups similar outputs, measures spread. | Multiple generations + NLI |
| **SemanticDensityMCMaskedNLL** | sample NLLs + semantic kernel | Weighted semantic kernel density. | Multiple generations + NLI |
| **MonteCarloSequenceEntropyMCMaskedNLL** | sample NLLs | Predictive entropy from MC samples. | Multiple generations |
| **MonteCarloNormalizedSequenceEntropyMCMaskedNLL** | sample normalized NLLs | Normalized predictive entropy. | Multiple generations |
| **SentenceSARMCMaskedNLL** | sample NLLs + sentence similarity | SAR score combining probability with text relevance. | Multiple generations |

---

## Method Landscape: Cost vs Signal Type

| Signal Type | Free (generation only) | Moderate (NLI model) | Expensive (extra generations) |
|------------|----------------------|---------------------|----------------------------|
| **Token-level** | TraceProbs, LastStepConfidence, Top2GapTrace | | MCMaskedNLL, BlockMCMaskedNLL |
| **Dynamics** | TraceNLLVariance, ConfidenceSlope, ConfidenceCurvature, ArgmaxFlipCount | | |
| **Convergence** | PhaseTransitionSharpness, TokenRankStability, MCMCDiagnostics | | |
| **Commitment** | ConfidenceAtCommitment, MaskNeighborRatio, CommitmentSynchrony | | |
| **Multi-scale** | BlockUncertaintyVariance, PositionConfidenceCorrelation | | |
| **Remasking** | RemaskCountStat, FirstRemaskStepCount, NFEUncertainty | | |
| **Lexical** | LexicalSimilarityTrace | | |
| **Semantic** | | DegMatTrace, EigValLaplacian, EccentricityTrace, NumSemSetsTrace, LUQTrace, KernelLanguageEntropy, AvgDissimilarityTrace | SemanticEntropy, SemanticDensity, SentenceSAR |
| **Composite** | | CocoaMCMaskedNLL(Trace) | |

---

## What Makes Diffusion LLM UQ Unique vs Autoregressive

Autoregressive models generate tokens left-to-right in one pass. Diffusion LLMs unmask tokens iteratively — the same token may be predicted, remasked, and re-predicted many times. This creates signals that **do not exist in AR models**:

1. **Temporal trajectory per token** — how a token's prediction evolves over multiple refinement steps (TraceNLLVariance, ArgmaxFlipCount, ConfidenceSlope)
2. **Commitment dynamics** — when and under what conditions each token gets "locked in" (ConfidenceAtCommitment, MaskNeighborRatio, FirstRemaskStepCount)
3. **Parallel context** — unlike AR where all left context is available, in dLLMs a token's context is partially masked and gradually revealed (MaskNeighborRatio)
4. **Global refinement** — the model can reconsider all tokens simultaneously, so uncertainty reflects global coherence not just local next-token prediction
5. **Block structure** — semi-autoregressive generation creates block-level signals not available in pure AR or pure diffusion
6. **Convergence properties** — the iterative process is analogous to MCMC sampling, enabling convergence diagnostics (R-hat, ESS) and phase transition analysis

---

## Research Directions

### A. Unexploited signals in existing traces

- **StepwiseKL / Entropy Change Rate**: How fast is the distribution changing between consecutive steps? Proxy: `|entropy[t+1] - entropy[t]|` averaged across steps. Rapid changes = unstable.
- **Confidence Quantile Features**: Instead of avg/max, use percentiles (p10, p50, p90) of per-token confidence. The shape of the confidence distribution may be more informative than simple aggregations.

### B. Structural / positional signals

- **PositionalUncertaintyProfile**: Does uncertainty correlate with token position? Beginning-of-answer vs end-of-answer uncertainty could signal different failure modes.
- **BlockBoundaryUncertainty**: For semi-autoregressive generation, are tokens at block boundaries more uncertain? The block transition is a structural artifact.
- **DependencyChainLength**: In the remasking graph, how long are the chains of tokens that influence each other's commitment? Long chains = cascading uncertainty.

### C. Distribution-level signals (require storing more data)

- **Full StepwiseKL**: KL divergence between consecutive steps' full distributions. Expensive to store (vocab_size per step) but most informative. Could approximate with top-k only.
- **Probability Mass Concentration**: At each step, what fraction of probability mass is in the top-k tokens? Low concentration = model is diffuse = uncertain.
- **Token Rank Trajectory (Option B)**: Store the log-prob of the *committed token* at every step (retrospective). Richer than TokenRankStability Option A (argmax match only) — captures the full probability trajectory of the winning token.

### D. Cross-token interaction signals

- **NeighborArgmaxCorrelation**: Do adjacent tokens tend to flip together or independently? Correlated flips suggest structural uncertainty; independent flips suggest local uncertainty.
- **MaskedContextQuality**: When a token is committed, what's the average confidence of its non-masked neighbors? Committed among confident neighbors = more reliable.
- **CommitmentOrderEntropy**: Is the order in which tokens get committed predictable or random? If the model commits tokens in an unusual order (e.g., function words after content words), the generation may be unreliable.
