from typing import Any, Dict

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .greedy_cross_encoder_similarity_block_trace import (
    GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator,
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

    This constructs our custom `GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator`
    using the provided config.
    """
    batch_size = _get_cfg_value(cfg, "batch_size", 10)
    cross_encoder_name = _get_cfg_value(
        cfg, "cross_encoder_name", "cross-encoder/stsb-roberta-large"
    )
    return GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator(
        batch_size=batch_size,
        cross_encoder_name=cross_encoder_name,
    )
