import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class TraceEntropy(Estimator):
    """
    Estimates sequence-level uncertainty via entropy from generation trace.

    For each example, the method:
      - Takes `gen_trace_entropy` which contains entropy values at each step
      - Aggregates across steps using the configured aggregation method (avg/max)
      - Returns the score as uncertainty (higher = more uncertain)
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(
            ["gen_trace_entropy", "greedy_tokens", "tokenizer"], "sequence"
        )
        self.aggregation = aggregation

    def __str__(self):
        return f"TraceEntropy_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_entropy = stats["gen_trace_entropy"]
        _, num_blocks, _, block_length = trace_entropy.shape
        max_token_count = num_blocks * block_length
        special_token_ids = set(
            getattr(stats["tokenizer"], "all_special_ids", [])
        )

        result = []
        for sample_idx, sample in enumerate(trace_entropy):
            token_ids = stats["greedy_tokens"][sample_idx]
            token_count = min(len(token_ids), max_token_count)
            while (
                token_count > 0
                and int(token_ids[token_count - 1]) in special_token_ids
            ):
                token_count -= 1

            position_mask = (
                torch.arange(max_token_count, device=sample.device) < token_count
            ).reshape(num_blocks, block_length)
            valid_mask = torch.isfinite(sample) & position_mask[:, None, :]
            step_counts = valid_mask.sum(dim=-1)
            step_sums = torch.where(
                valid_mask, sample, torch.zeros_like(sample)
            ).sum(dim=-1)
            step_means = step_sums / step_counts.clamp_min(1)
            valid_values = step_means[step_counts > 0]

            if valid_values.numel() == 0:
                result.append(0.0)
            elif self.aggregation == "avg":
                result.append(valid_values.mean().item())
            elif self.aggregation == "max":
                result.append(valid_values.max().item())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Return entropy as uncertainty (higher = more uncertain)
        return np.asarray(result, dtype=np.float64)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a TraceEntropy estimator from a config dictionary.
    """
    aggregation = cfg.get("aggregation", "avg")
    return TraceEntropy(aggregation=aggregation)
