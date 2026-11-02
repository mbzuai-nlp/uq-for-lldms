import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class NFEUncertainty(Estimator):
    """
    Estimates sequence-level uncertainty using the number of function evaluations (NFE)
    from diffusion model generation. Higher NFE indicates more uncertainty as it means
    the model needed more steps to generate the sequence.
    """

    def __init__(self):
        super().__init__(["nfe"], "sequence")

    def __str__(self):
        return "NFEUncertainty"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns NFE as uncertainty score.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics containing:
                * 'nfe': number of function evaluations (scalar for batch or per-sample)
        Returns:
            np.ndarray: NFE values as uncertainty scores. Higher values indicate more uncertainty.
        """
        nfe = stats["nfe"]
        return np.array(nfe)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `NFEUncertainty` estimator from a config dictionary.
    """
    return NFEUncertainty()
