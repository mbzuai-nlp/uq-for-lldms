import numpy as np
import torch

from typing import Dict, List, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class DiffusionGemmaGreedyProbsCalculator(StatCalculator):
    """Aggregate-only greedy-probs calculator for DiffusionGemma.

    Modelled on :class:`DiffusionGreedyProbsCalculator` but adapted to the fact that
    DiffusionGemma ``generate()`` returns ``scores=None`` and uses a 256k-token vocab:

      * NEVER calls ``torch.stack(out.scores)`` — there are no AR scores.
      * Reads ``greedy_log_likelihoods`` (the diffusion-native committed-token log-prob)
        directly off the backend output object, not from scores.
      * DROPS the full ``greedy_log_probs`` (256k vocab ≈ 131 MB/sample), the top-k
        ``greedy_tokens_alternatives``, and all embeddings/attention stats.
      * ``store_full_step_logits`` must stay ``False`` (materialising per-step canvas
        logits over the 256k vocab OOMs).

    The emitted stat keys are exactly those consumed by the DiffusionGemma estimator
    set (TraceEntropy/TraceProbs, ConfidenceAtCommit, LUQ/KLE-trace via the kept
    SemanticMatrix/CrossEncoder/Deberta trace calculators, and D-CoCoA-L).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "input_texts",
            "input_tokens",
            "greedy_tokens",
            "greedy_texts",
            "greedy_log_likelihoods",
            "tokenizer",
            "nfe",
            "remask_count",
            "first_remask_step",
            "generation_trace",
            "full_generation_trace",
            "last_block_generation_trace",
            "block_generation_traces",
            "gen_trace_logits",
            "gen_trace_entropy",
            "gen_trace_argmax",
            "gen_trace_top2gap",
            "confidence_at_commit",
            "mask_fraction_at_commit",
        ], []

    def __init__(
        self,
        output_diffusion_stats: bool = True,
        store_full_step_logits: bool = False,
        n_alternatives: int = 10,
    ):
        super().__init__()
        self.output_diffusion_stats = output_diffusion_stats
        # Guard: keeping full per-step canvas logits over the 256k vocab OOMs.
        self.store_full_step_logits = bool(store_full_step_logits)
        self.n_alternatives = n_alternatives

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                num_return_sequences=1,
            )
        # DiffusionGemma returns scores=None: we never touch out.scores and never
        # torch.stack(out.scores). greedy_log_likelihoods is synthesized by the backend
        # (per-position committed-token log-prob).

        result_dict = {
            "input_texts": list(texts),
            "input_tokens": batch["input_ids"].detach().to("cpu").tolist(),
            "greedy_tokens": out.greedy_tokens,
            "greedy_texts": out.greedy_texts,
            "greedy_log_likelihoods": out.greedy_log_likelihoods,
            "tokenizer": model.tokenizer,
        }

        if self.output_diffusion_stats:
            result_dict.update(
                {
                    "nfe": out.nfe,
                    "remask_count": out.remask_count,
                    "first_remask_step": out.first_remask_step,
                    "generation_trace": out.generation_trace,
                    "full_generation_trace": out.full_generation_trace,
                    "last_block_generation_trace": out.last_block_generation_trace,
                    "block_generation_traces": out.block_generation_traces,
                    "gen_trace_logits": out.gen_trace_logits,
                    "gen_trace_entropy": out.gen_trace_entropy,
                    "gen_trace_argmax": out.gen_trace_argmax,
                    "gen_trace_top2gap": out.gen_trace_top2gap,
                    "confidence_at_commit": out.confidence_at_commit,
                    "mask_fraction_at_commit": out.mask_fraction_at_commit,
                }
            )

        return result_dict
