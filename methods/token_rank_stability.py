import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class TokenRankStability(Estimator):
    """
    Estimates sequence-level uncertainty by measuring how consistently
    the final committed token was the model's top prediction across
    diffusion steps.

    A token that was always #1 (argmax consistency = 1.0) is genuinely
    certain. A token that only became #1 at the last step (consistency
    near 0.0) may have misleadingly high final confidence — the model
    changed its mind dramatically.

    This separates "confident now" from "has always been confident about
    this specific token" — a distinction unique to iterative generation.

    Variants:
    - "consistency": fraction of steps where argmax matches the final
      committed token (higher = more certain)
    - "first_appearance": step at which the committed token first becomes
      the argmax, as a fraction of total steps (later = more uncertain)
    """

    def __init__(self, variant: str = "consistency", aggregation: str = "avg"):
        super().__init__(["gen_trace_argmax"], "sequence")
        self.variant = variant
        self.aggregation = aggregation

    def __str__(self):
        return f"RankStability_{self.variant}_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_argmax = stats["gen_trace_argmax"]  # (B, blocks, steps, block_len), padded -1

        B = trace_argmax.shape[0]
        result = []

        for i in range(B):
            # (num_tokens, max_steps)
            argmax_t = trace_argmax[i].permute(0, 2, 1).reshape(-1, trace_argmax.shape[2])

            token_scores = []
            for t in range(argmax_t.shape[0]):
                steps = argmax_t[t]
                valid = steps[steps != -1]

                if len(valid) < 1:
                    token_scores.append(0.0)
                    continue

                final_token = valid[-1].item()

                if self.variant == "consistency":
                    # Fraction of steps where argmax == final token
                    matches = (valid == final_token).float().mean().item()
                    token_scores.append(matches)

                elif self.variant == "first_appearance":
                    # Step where final token first appears as argmax (fraction)
                    match_indices = (valid == final_token).nonzero()
                    if len(match_indices) > 0:
                        first_idx = match_indices[0].item()
                        token_scores.append(first_idx / len(valid))
                    else:
                        token_scores.append(1.0)

                else:
                    raise ValueError(f"Unknown variant: {self.variant}")

            scores = np.array(token_scores)

            if self.aggregation == "avg":
                result.append(scores.mean())
            elif self.aggregation == "min":
                result.append(scores.min())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        result = np.array(result)

        if self.variant == "consistency":
            # Negate: high consistency = more certain = lower uncertainty
            return -result
        elif self.variant == "first_appearance":
            # Direct: late first appearance = more uncertain
            return result
        return result


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "consistency")
    aggregation = cfg.get("aggregation", "avg")
    return TokenRankStability(variant=variant, aggregation=aggregation)
