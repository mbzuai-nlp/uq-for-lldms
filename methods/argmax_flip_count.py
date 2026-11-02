import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class ArgmaxFlipCount(Estimator):
    """
    Estimates sequence-level uncertainty by counting how many times the
    predicted token identity changes across diffusion steps.

    A token that keeps switching between candidates ("72" -> "70" -> "76")
    across steps is uncertain. A token that stays as "Paris" throughout
    is certain.

    Uses `gen_trace_argmax` which contains the argmax token ID at each
    diffusion step. Counts consecutive changes per token, then aggregates
    to sequence level.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(
            ["gen_trace_argmax", "greedy_tokens", "tokenizer"], "sequence"
        )
        self.aggregation = aggregation

    def __str__(self):
        return f"ArgmaxFlipCount_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_argmax = stats["gen_trace_argmax"]

        # trace_argmax: (B, num_blocks, max_steps, block_length), dtype long, padded with -1
        B = trace_argmax.shape[0]
        max_token_count = trace_argmax.shape[1] * trace_argmax.shape[3]
        special_token_ids = set(
            getattr(stats["tokenizer"], "all_special_ids", [])
        )

        result = []
        for i in range(B):
            # (num_blocks, max_steps, block_length)
            sample = trace_argmax[i]
            # Reshape to (num_tokens, max_steps)
            # permute to (num_blocks, block_length, max_steps) then flatten first two
            sample_t = sample.permute(0, 2, 1)  # (num_blocks, block_length, max_steps)
            flat = sample_t.reshape(-1, sample_t.shape[-1])  # (num_tokens, max_steps)

            token_ids = stats["greedy_tokens"][i]
            token_count = min(len(token_ids), max_token_count)
            while (
                token_count > 0
                and int(token_ids[token_count - 1]) in special_token_ids
            ):
                token_count -= 1
            flat = flat[:token_count]

            token_flips = []
            for t in range(flat.shape[0]):
                steps = flat[t]
                valid_mask = steps != -1
                valid_steps = steps[valid_mask]
                if len(valid_steps) >= 2:
                    # Count consecutive differences
                    flips = (valid_steps[1:] != valid_steps[:-1]).sum().item()
                    token_flips.append(flips)

            if not token_flips:
                result.append(0.0)
                continue

            token_flips = np.asarray(token_flips, dtype=np.float64)
            if self.aggregation == "avg":
                result.append(token_flips.mean())
            elif self.aggregation == "max":
                result.append(token_flips.max())
            elif self.aggregation == "sum":
                result.append(token_flips.sum())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return ArgmaxFlipCount(aggregation=aggregation)
