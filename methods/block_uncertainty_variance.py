import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class BlockUncertaintyVariance(Estimator):
    """
    Estimates sequence-level uncertainty via the variance of uncertainty
    across semi-autoregressive generation blocks.

    Some blocks may be easy (high confidence) while others are hard
    (low confidence). High block-to-block variance means the model's
    difficulty is uneven — some parts of the answer are much harder
    than others, which can indicate partial knowledge or structural
    confusion.

    This directly leverages the block generation structure unique to
    semi-autoregressive diffusion LLMs.

    Variants:
    - "logprob": uses per-block mean log-prob as the block-level score
    - "entropy": uses per-block mean entropy as the block-level score

    Returns the variance (or range) across blocks.
    """

    def __init__(self, variant: str = "logprob", aggregation: str = "var"):
        deps = ["gen_trace_logits"] if variant == "logprob" else ["gen_trace_entropy"]
        super().__init__(deps, "sequence")
        self.variant = variant
        self.aggregation = aggregation

    def __str__(self):
        return f"BlockUncVar_{self.variant}_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if self.variant == "logprob":
            trace = stats["gen_trace_logits"]  # (B, num_blocks, max_steps, block_length)
        else:
            trace = stats["gen_trace_entropy"]

        B = trace.shape[0]
        num_blocks = trace.shape[1]

        result = []
        for i in range(B):
            block_scores = []
            for b in range(num_blocks):
                # (max_steps, block_length) for this block
                block_data = trace[i, b]

                # Flatten and filter valid values
                flat = block_data.reshape(-1)
                valid = flat[torch.isfinite(flat)]

                if len(valid) > 0:
                    block_scores.append(valid.mean().item())
                # Skip blocks with no valid data (fully padded)

            if len(block_scores) < 2:
                result.append(0.0)
                continue

            scores = np.array(block_scores)

            if self.aggregation == "var":
                result.append(scores.var())
            elif self.aggregation == "range":
                result.append(scores.max() - scores.min())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Direct: high variance across blocks = uneven difficulty = uncertain
        return np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "logprob")
    aggregation = cfg.get("aggregation", "var")
    return BlockUncertaintyVariance(variant=variant, aggregation=aggregation)
