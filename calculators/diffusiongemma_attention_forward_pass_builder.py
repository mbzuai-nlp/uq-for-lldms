from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .diffusiongemma_attention_forward_pass import DiffusionGemmaAttentionForwardPassCalculator


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    return DiffusionGemmaAttentionForwardPassCalculator()
