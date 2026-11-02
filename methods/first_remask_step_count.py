import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class FirstRemaskStepCount(Estimator):
    """
    Estimates sequence-level uncertainty using the count of -1 values in first_remask_step.
    The first_remask_step indicates the step at which each token was first transferred (-1 if never transferred).
    Less -1 values (never transferred tokens) indicate more uncertainty.
    """

    def __init__(self, aggregation: str = "count"):
        super().__init__(["first_remask_step"], "sequence")
        self.aggregation = aggregation

    def __str__(self):
        return f"FirstRemaskStep{self.aggregation.capitalize()}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns negative count of -1 values in first_remask_step as uncertainty score.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics containing:
                * 'first_remask_step': tensor of shape (B, L) with first remask step for each token
                  (-1 if token was never transferred)
        Returns:
            np.ndarray: Negative count of -1 values per sample as uncertainty scores.
                Higher values indicate more uncertainty (more tokens never transferred).
        """
        first_remask_step = stats["first_remask_step"]
        if self.aggregation == "count":
            return np.array(
                [-np.sum(np.asarray(frs) <= 0) for frs in first_remask_step]
            )
        elif self.aggregation == "mean_step":
            return np.array(
                [-frs.numel() / frs.unique().numel() for frs in first_remask_step]
            )


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `FirstRemaskStepCount` estimator from a config dictionary.
    """
    aggregation = cfg.get("aggregation", "count")
    return FirstRemaskStepCount(aggregation=aggregation)
