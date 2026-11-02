from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .greedy_cross_encoder_similarity_trace import (
    GreedyCrossEncoderSimilarityMatrixTraceCalculator,
)


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """
    Safely get a value from an OmegaConf node or a plain dict.
    """
    if cfg is None:
        return default
    # OmegaConf nodes support attribute-style access; dicts support get()
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    """
    Builder entrypoint compatible with `FactoryStatCalculator`.

    This does not touch `lm_polygraph` internals; it simply constructs
    our custom `GreedyCrossEncoderSimilarityMatrixTraceCalculator` using the provided config.
    """
    return GreedyCrossEncoderSimilarityMatrixTraceCalculator()
