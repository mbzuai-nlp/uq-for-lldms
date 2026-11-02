import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class MaskNeighborRatio(Estimator):
    """
    Estimates sequence-level uncertainty using the fraction of the sequence
    that was still masked when each token was committed.

    Tokens committed when most of the sequence is still masked had less
    context available — the model was making decisions with incomplete
    information. Higher mask fraction at commitment = more uncertain.

    This is unique to diffusion LLMs — it exploits the masked diffusion
    process where tokens are revealed gradually. No autoregressive
    equivalent exists.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["mask_fraction_at_commit"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"MaskNeighborRatio_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # (B, gen_length) — fraction of sequence masked at commitment, -inf if never committed
        mfrac = stats["mask_fraction_at_commit"]

        result = []
        for i in range(mfrac.shape[0]):
            valid_mask = torch.isfinite(mfrac[i])
            valid = mfrac[i][valid_mask]
            if len(valid) > 0:
                if self.aggregation == "avg":
                    result.append(valid.mean().item())
                elif self.aggregation == "max":
                    result.append(valid.max().item())
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            else:
                result.append(0.0)

        # Higher mask fraction = more uncertain (direct, no negation)
        return np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return MaskNeighborRatio(aggregation=aggregation)
