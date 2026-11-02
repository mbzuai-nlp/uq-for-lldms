import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class Top2GapTrace(Estimator):
    """
    Estimates sequence-level uncertainty via the gap between the top-2
    predicted token probabilities across diffusion steps.

    A small gap means the model is torn between two candidates (uncertain).
    A large gap means a clear winner (certain).

    Uses `gen_trace_top2gap` which contains (p_1st - p_2nd) per token at
    each diffusion step. Aggregates across steps and tokens.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["gen_trace_top2gap"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"Top2GapTrace_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_top2gap = stats["gen_trace_top2gap"]

        # (B, num_blocks, max_steps, block_length)
        B = trace_top2gap.shape[0]

        result = []
        for i in range(B):
            sample = trace_top2gap[i]
            # (num_tokens, max_steps)
            sample_t = sample.permute(0, 2, 1).reshape(-1, sample.shape[1])

            token_gaps = []
            for t in range(sample_t.shape[0]):
                valid_mask = torch.isfinite(sample_t[t])
                valid = sample_t[t][valid_mask]
                if len(valid) > 0:
                    if self.aggregation == "avg":
                        token_gaps.append(valid.mean().item())
                    elif self.aggregation == "min":
                        token_gaps.append(valid.min().item())
                    else:
                        raise ValueError(f"Unknown aggregation: {self.aggregation}")
                else:
                    token_gaps.append(0.0)

            scores = np.array(token_gaps)
            result.append(scores.mean())

        # Negate: small gap = more uncertain
        return -np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return Top2GapTrace(aggregation=aggregation)
