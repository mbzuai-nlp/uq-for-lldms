import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class ConfidenceSlope(Estimator):
    """
    Estimates sequence-level uncertainty via the trend of max log-probs
    across diffusion steps.

    Negative slope = confidence is decreasing as generation progresses
    (model is becoming less sure with more context — bad sign).
    Positive slope = confidence is increasing (model is converging).

    Uses a simple (last - first) / num_steps approximation per token,
    then aggregates to sequence level.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["gen_trace_logits"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"ConfidenceSlope_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_logits = stats["gen_trace_logits"]

        # (B, num_blocks, max_steps, block_length)
        B = trace_logits.shape[0]

        result = []
        for i in range(B):
            sample = trace_logits[i]
            # (num_blocks, block_length, max_steps)
            sample_t = sample.permute(0, 2, 1)
            # (num_tokens, max_steps)
            flat = sample_t.reshape(-1, sample_t.shape[-1])

            token_slopes = []
            for t in range(flat.shape[0]):
                valid_mask = torch.isfinite(flat[t])
                valid_values = flat[t][valid_mask]
                if len(valid_values) >= 2:
                    # (last - first) / (num_steps - 1)
                    slope = (valid_values[-1] - valid_values[0]).item() / (len(valid_values) - 1)
                    token_slopes.append(slope)
                else:
                    token_slopes.append(0.0)

            token_slopes = np.array(token_slopes)

            if self.aggregation == "avg":
                result.append(token_slopes.mean())
            elif self.aggregation == "min":
                result.append(token_slopes.min())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Negate: negative slope (declining confidence) = more uncertain
        return -np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return ConfidenceSlope(aggregation=aggregation)
