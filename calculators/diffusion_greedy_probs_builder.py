from typing import Any

from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .diffusion_greedy_probs import DiffusionGreedyProbsCalculator


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
    our custom `DiffusionGreedyProbsCalculator` using the provided config.
    """
    output_attentions = _get_cfg_value(cfg, "output_attentions", True)
    output_hidden_states = _get_cfg_value(cfg, "output_hidden_states", True)
    output_hidden_states_trace = _get_cfg_value(cfg, "output_hidden_states_trace", False)
    output_all_hidden_states_trace = _get_cfg_value(
        cfg, "output_all_hidden_states_trace", False
    )
    output_diffusion_stats = _get_cfg_value(cfg, "output_diffusion_stats", True)
    output_token_embeddings = _get_cfg_value(cfg, "output_token_embeddings", False)
    hidden_state_layer = _get_cfg_value(cfg, "hidden_state_layer", "middle")
    return DiffusionGreedyProbsCalculator(
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        output_diffusion_stats=output_diffusion_stats,
        output_token_embeddings=output_token_embeddings,
        hidden_state_layer=hidden_state_layer,
        output_hidden_states_trace=output_hidden_states_trace,
        output_all_hidden_states_trace=output_all_hidden_states_trace,
    )
