from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)

from .safe_lookback_ratio import SafeLookbackRatioCalculator


def load_stat_calculator(cfg: Any, env: BuilderEnvironmentBase):
    """Build the project-local lookback calculator without patching lm-polygraph."""
    env.return_lookback_ratios = True
    eps = getattr(cfg, "eps", 1e-6) if cfg is not None else 1e-6
    return SafeLookbackRatioCalculator(eps=float(eps))
