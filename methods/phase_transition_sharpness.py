import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class PhaseTransitionSharpness(Estimator):
    """
    Estimates sequence-level uncertainty via the sharpness of each token's
    "crystallization" — the transition from uncertain to committed.

    For each token, finds the stabilization point (last argmax flip) and
    measures the confidence jump at that moment. Sharp transitions (large
    sudden jumps) combined with high final confidence indicate genuine
    certainty. Gradual transitions or forced commitments indicate
    uncertainty.

    Variants:
    - "jump": magnitude of log-prob increase at stabilization step
    - "width": number of steps for log-prob to go from 20% to 80% of
      final value (fewer = sharper). Returned as negative (wider = more uncertain).
    - "late_stability": fraction of total steps after which the token
      stabilizes (later = more uncertain)
    """

    def __init__(self, variant: str = "jump", aggregation: str = "avg"):
        super().__init__(["gen_trace_logits", "gen_trace_argmax"], "sequence")
        self.variant = variant
        self.aggregation = aggregation

    def __str__(self):
        return f"PhaseTransition_{self.variant}_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        trace_logits = stats["gen_trace_logits"]   # (B, blocks, steps, block_len)
        trace_argmax = stats["gen_trace_argmax"]   # same shape, long, padded -1

        B = trace_logits.shape[0]
        result = []

        for i in range(B):
            # Reshape to (num_tokens, max_steps)
            logits_t = trace_logits[i].permute(0, 2, 1).reshape(-1, trace_logits.shape[2])
            argmax_t = trace_argmax[i].permute(0, 2, 1).reshape(-1, trace_argmax.shape[2])

            token_scores = []
            for t in range(logits_t.shape[0]):
                lp = logits_t[t]
                am = argmax_t[t]

                valid_mask = torch.isfinite(lp) & (am != -1)
                valid_lp = lp[valid_mask]
                valid_am = am[valid_mask]

                if len(valid_am) < 2:
                    token_scores.append(0.0)
                    continue

                # Find stabilization point: last step where argmax changes
                flips = (valid_am[1:] != valid_am[:-1])
                if flips.any():
                    # Last flip index (in valid_am space)
                    last_flip = flips.nonzero()[-1].item() + 1
                else:
                    # Never flipped — fully stable from step 0
                    last_flip = 0

                n_valid = len(valid_lp)

                if self.variant == "jump":
                    # Absolute confidence jump at stabilization step
                    # Never flipped: no transition needed, assign 0 (no uncertainty from transition)
                    # Flipped: measure how abrupt the final commitment was
                    if last_flip == 0:
                        jump = 0.0
                    elif last_flip < n_valid:
                        jump = abs((valid_lp[last_flip] - valid_lp[last_flip - 1]).item())
                    else:
                        jump = 0.0
                    token_scores.append(jump)

                elif self.variant == "width":
                    # Transition width: steps to go from 20% to 80% of final value
                    final_val = valid_lp[-1].item()
                    first_val = valid_lp[0].item()
                    if abs(final_val - first_val) < 1e-6:
                        token_scores.append(0.0)
                        continue
                    thresh_20 = first_val + 0.2 * (final_val - first_val)
                    thresh_80 = first_val + 0.8 * (final_val - first_val)
                    step_20 = 0
                    step_80 = n_valid - 1
                    for s in range(n_valid):
                        if valid_lp[s].item() >= thresh_20:
                            step_20 = s
                            break
                    for s in range(n_valid):
                        if valid_lp[s].item() >= thresh_80:
                            step_80 = s
                            break
                    width = max(step_80 - step_20, 1)
                    token_scores.append(float(width))

                elif self.variant == "late_stability":
                    # Fraction of steps after which token stabilizes
                    # Later stabilization = more uncertain
                    token_scores.append(last_flip / n_valid)

                else:
                    raise ValueError(f"Unknown variant: {self.variant}")

            scores = np.array(token_scores)

            if self.aggregation == "avg":
                result.append(scores.mean())
            elif self.aggregation == "max":
                result.append(scores.max())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        result = np.array(result)

        if self.variant == "jump":
            # Direct: large jump at commitment = abrupt transition = more uncertain
            return result
        elif self.variant == "width":
            # Direct: wider transition = more uncertain
            return result
        elif self.variant == "late_stability":
            # Direct: later stability = more uncertain
            return result
        return result


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "jump")
    aggregation = cfg.get("aggregation", "avg")
    return PhaseTransitionSharpness(variant=variant, aggregation=aggregation)
