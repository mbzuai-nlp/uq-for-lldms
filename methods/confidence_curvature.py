import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class ConfidenceCurvature(Estimator):
    """
    Estimates sequence-level uncertainty via the second derivative
    (curvature) of max log-probs across diffusion steps.

    While ConfidenceSlope captures the linear trend (is confidence going
    up or down?), curvature captures nonlinear dynamics:
    - High curvature = the model is "backtracking" or over-correcting
      between steps — confidence accelerates/decelerates sharply.
    - Low curvature = smooth, steady convergence.

    This detects failure modes where confidence is oscillating or
    making abrupt corrections that first-order slope misses.

    Computed as: mean |logprob[t+1] - 2*logprob[t] + logprob[t-1]|
    across valid steps per token, then aggregated to sequence level.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["gen_trace_logits"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"ConfidenceCurvature_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_logits = stats["gen_trace_logits"]

        # (B, num_blocks, max_steps, block_length)
        B = trace_logits.shape[0]

        result = []
        for i in range(B):
            sample = trace_logits[i]
            # (num_tokens, max_steps)
            sample_t = sample.permute(0, 2, 1)
            sample_t = sample_t.reshape(-1, sample_t.shape[-1])

            token_curvatures = []
            for t in range(sample_t.shape[0]):
                valid_mask = torch.isfinite(sample_t[t])
                valid = sample_t[t][valid_mask]

                if len(valid) >= 3:
                    # Second finite difference: f[t+1] - 2*f[t] + f[t-1]
                    second_diff = valid[2:] - 2 * valid[1:-1] + valid[:-2]
                    curvature = second_diff.abs().mean().item()
                    token_curvatures.append(curvature)
                else:
                    # Not enough points for second derivative
                    token_curvatures.append(0.0)

            scores = np.array(token_curvatures)

            if self.aggregation == "avg":
                result.append(scores.mean())
            elif self.aggregation == "max":
                result.append(scores.max())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Direct: high curvature = more uncertain
        return np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return ConfidenceCurvature(aggregation=aggregation)
