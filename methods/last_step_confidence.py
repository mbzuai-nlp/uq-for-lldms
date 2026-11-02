import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class LastStepConfidence(Estimator):
    """
    Estimates sequence-level uncertainty using only the max log-prob from
    the final diffusion step (when the token is committed).

    Unlike TraceProbs which averages across all steps, this uses only the
    last step which has the most context (fewest masked tokens remaining)
    and represents the model's most informed prediction.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["gen_trace_logits"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"LastStepConfidence_{self.aggregation}"

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

            token_last_conf = []
            for t in range(flat.shape[0]):
                valid_mask = torch.isfinite(flat[t])
                valid_values = flat[t][valid_mask]
                if len(valid_values) >= 1:
                    token_last_conf.append(valid_values[-1].item())
                else:
                    token_last_conf.append(float('-inf'))

            token_last_conf = np.array(token_last_conf)
            valid = token_last_conf[np.isfinite(token_last_conf)]

            if len(valid) > 0:
                if self.aggregation == "avg":
                    result.append(valid.mean())
                elif self.aggregation == "min":
                    result.append(valid.min())
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            else:
                result.append(float('-inf'))

        # Negate: low confidence = more uncertain
        return -np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return LastStepConfidence(aggregation=aggregation)
