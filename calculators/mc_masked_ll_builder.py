from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .mc_masked_ll_calculator import MCMaskedLLCalculator


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
    our custom `MCMaskedLLCalculator` using the provided config.
    """
    mc_num = _get_cfg_value(cfg, "mc_num", 16)
    mc_batch_size = _get_cfg_value(cfg, "mc_batch_size", 16)
    return MCMaskedLLCalculator(
        mc_num=mc_num, mc_batch_size=mc_batch_size
    )
