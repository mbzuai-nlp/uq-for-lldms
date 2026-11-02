from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .diffusiongemma_sample import DiffusionGemmaSamplingGenerationCalculator


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """Safely get a value from an OmegaConf node or a plain dict."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    """Builder entrypoint compatible with ``FactoryStatCalculator``.

    Constructs :class:`DiffusionGemmaSamplingGenerationCalculator`, the DiffusionGemma
    replacement for lm-polygraph's ``SamplingGenerationCalculator``. ``samples_n`` sets how
    many decodes are drawn per input and is the dominant cost of the sampling-based
    estimators: each sample is a full diffusion generation, so the run is roughly
    ``samples_n`` times slower than the greedy-only sweep.
    """
    samples_n = _get_cfg_value(cfg, "samples_n", 10)
    sample_temperature = _get_cfg_value(cfg, "sample_temperature", None)
    warn_on_degenerate = _get_cfg_value(cfg, "warn_on_degenerate", True)
    return DiffusionGemmaSamplingGenerationCalculator(
        samples_n=samples_n,
        sample_temperature=sample_temperature,
        warn_on_degenerate=warn_on_degenerate,
    )
