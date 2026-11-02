import numpy as np
import torch
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class ConfidenceAtCommitment(Estimator):
    """
    Estimates sequence-level uncertainty using the log-prob at the exact
    step each token is committed (no longer remasked).

    Unlike LastStepConfidence (which uses the last step of the block),
    this captures the confidence at each token's individual decision
    point. A token committed early with high confidence is certain;
    a token committed late with low confidence is uncertain.

    This is unique to diffusion LLMs — no autoregressive equivalent exists
    since AR models commit tokens in a single pass.
    """

    def __init__(self, aggregation: str = "avg"):
        super().__init__(["confidence_at_commit"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"ConfidenceAtCommit_{self.aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # (B, gen_length) — log-prob at commitment, -inf if never committed
        conf = stats["confidence_at_commit"]

        result = []
        for i in range(conf.shape[0]):
            valid_mask = torch.isfinite(conf[i])
            valid = conf[i][valid_mask]
            if len(valid) > 0:
                if self.aggregation == "avg":
                    result.append(valid.mean().item())
                elif self.aggregation == "min":
                    result.append(valid.min().item())
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            else:
                result.append(float('-inf'))

        # Negate: low confidence = more uncertain
        return -np.array(result)


def load_estimator(cfg: Dict) -> Estimator:
    aggregation = cfg.get("aggregation", "avg")
    return ConfidenceAtCommitment(aggregation=aggregation)
