from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .diffusiongemma_greedy_probs import DiffusionGemmaGreedyProbsCalculator


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """Safely get a value from an OmegaConf node or a plain dict."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    """Builder entrypoint compatible with ``FactoryStatCalculator``.

    Constructs the aggregate-only :class:`DiffusionGemmaGreedyProbsCalculator`. Attentions
    and hidden states are never requested (Gemma's 256k vocab / large canvas make full
    logits and embeddings prohibitively expensive); ``store_full_step_logits`` defaults to
    ``False`` and should stay that way.
    """
    output_diffusion_stats = _get_cfg_value(cfg, "output_diffusion_stats", True)
    store_full_step_logits = _get_cfg_value(cfg, "store_full_step_logits", False)
    return DiffusionGemmaGreedyProbsCalculator(
        output_diffusion_stats=output_diffusion_stats,
        store_full_step_logits=store_full_step_logits,
    )
