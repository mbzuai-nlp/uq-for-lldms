import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class TraceNLLVariance(Estimator):
    """
    Estimates sequence-level uncertainty via variance of max log-probs across
    diffusion steps.

    A token with stable confidence across steps (low variance) is certain.
    A token whose confidence oscillates (high variance) is uncertain.

    Uses `gen_trace_logits` which contains max log-prob per token at each
    diffusion step. Computes per-token variance across steps, then aggregates
    to sequence level.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["gen_trace_logits"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"TraceNLLVariance_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_logits = stats["gen_trace_logits"]

        # trace_logits: (B, num_blocks, max_steps, block_length)
        B = trace_logits.shape[0]

        result = []
        for i in range(B):
            # (num_blocks, max_steps, block_length)
            sample = trace_logits[i]
            # Flatten blocks into token positions: (num_blocks * block_length, max_steps)
            # Reshape to (num_blocks, block_length, max_steps), then flatten first two dims
            sample_t = sample.permute(0, 2, 1)  # (num_blocks, block_length, max_steps)
            flat = sample_t.reshape(-1, sample_t.shape[-1])  # (num_tokens, max_steps)

            token_variances = []
            for t in range(flat.shape[0]):
                valid_mask = torch.isfinite(flat[t])
                valid_values = flat[t][valid_mask]
                if len(valid_values) >= 2:
                    token_variances.append(valid_values.var().item())
                else:
                    token_variances.append(0.0)

            token_variances = np.array(token_variances)

            if self.aggregation == "avg":
                result.append(token_variances.mean())
            elif self.aggregation == "max":
                result.append(token_variances.max())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return TraceNLLVariance(aggregation=aggregation)
