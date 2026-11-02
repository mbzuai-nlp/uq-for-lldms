import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class RemaskCountStat(Estimator):
    """
    Estimates sequence-level uncertainty using an aggregation (mean, max, or sum)
    of the remask count from diffusion model generation.
    The remask count indicates how many times each token was transferred during generation.
    Higher aggregate values indicate more uncertainty (depending on aggregation).
    """

    def __init__(self, aggregation: str = "mean"):
        """
        Args:
            aggregation (str): Aggregation strategy. One of "mean", "max", "sum".
                               Default is "mean".
        """
        super().__init__(["remask_count"], "sequence")
        if aggregation not in {"mean", "max", "sum"}:
            raise ValueError(f"Aggregation {aggregation} not supported")
        self.aggregation = aggregation
        self.agg_func = {"mean": np.mean, "max": np.max, "sum": np.sum}[
            self.aggregation
        ]

    def __str__(self):
        return f"RemaskCount{self.aggregation.capitalize()}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns aggregated remask count as uncertainty score.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics containing:
                * 'remask_count': tensor of shape (B, L) with remask counts for each token
        Returns:
            np.ndarray: Aggregated remask count per sample as uncertainty scores.
                Higher values indicate more uncertainty.
        """
        remask_count = stats["remask_count"]
        return np.array([self.agg_func(np.asarray(rc)) for rc in remask_count])


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `RemaskCountStat` estimator from a config dictionary.

    Config example:
        aggregation: "mean"  # or "max", "sum"
    """
    aggregation = cfg.get("aggregation", "mean")
    return RemaskCountStat(aggregation)
