import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class PositionConfidenceCorrelation(Estimator):
    """
    Estimates sequence-level uncertainty by measuring the correlation
    between token position and confidence.

    A strong negative correlation (later tokens less confident) may
    indicate the model is "losing track" as generation progresses.
    A strong positive correlation (later tokens more confident) is
    typical for semi-autoregressive generation where later blocks
    benefit from more revealed context.

    The absolute correlation is used as the uncertainty signal:
    high |correlation| = systematic positional bias in confidence,
    suggesting the model's certainty is driven by position rather
    than actual knowledge.

    Inspired by AlphaFold's per-residue confidence (pLDDT) which shows
    position-dependent uncertainty patterns in protein structure
    prediction.

    Variants:
    - "spearman": Spearman rank correlation (robust to outliers)
    - "pearson": Pearson linear correlation
    """

    def __init__(self, variant: str = "spearman"):
        super().__init__(["confidence_at_commit"], "sequence")
        self.variant = variant

    def __str__(self):
        return f"PosConfCorr_{self.variant}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        conf = stats["confidence_at_commit"]  # (B, gen_length)

        B = conf.shape[0]
        result = []

        for i in range(B):
            values = conf[i]
            valid_mask = torch.isfinite(values)
            valid = values[valid_mask].numpy()
            positions = torch.arange(len(values))[valid_mask].numpy()

            if len(valid) < 3:
                result.append(0.0)
                continue

            if self.variant == "spearman":
                from scipy.stats import spearmanr
                corr, _ = spearmanr(positions, valid)
            elif self.variant == "pearson":
                corr = np.corrcoef(positions, valid)[0, 1]
            else:
                raise ValueError(f"Unknown variant: {self.variant}")

            # Handle NaN from constant arrays
            if np.isnan(corr):
                corr = 0.0

            result.append(corr)

        result = np.array(result)

        # Negate: positive correlation (later tokens more confident) = more certain
        # Negative correlation (later tokens less confident) = more uncertain
        return -result


def load_estimator(cfg: Dict) -> Estimator:
    variant = cfg.get("variant", "spearman")
    return PositionConfidenceCorrelation(variant=variant)
