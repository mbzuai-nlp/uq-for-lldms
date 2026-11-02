from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .block_mc_masked_ll_calculator import BlockMCMaskedLLCalculator


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
    our custom `BlockMCMaskedLLCalculator` using the provided config.
    """
    block_size = _get_cfg_value(cfg, "block_size", 32)
    mc_num = _get_cfg_value(cfg, "mc_num", 16)
    seed = _get_cfg_value(cfg, "seed", 42)
    mc_batch_size = _get_cfg_value(cfg, "mc_batch_size", 16)
    return BlockMCMaskedLLCalculator(
        block_size=block_size,
        mc_num=mc_num,
        seed=seed,
        mc_batch_size=mc_batch_size,
    )
