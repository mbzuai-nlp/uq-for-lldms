from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentBase,
)
from .diffusiongemma_lm_probs import DiffusionGemmaGreedyLMProbsCalculator


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    return DiffusionGemmaGreedyLMProbsCalculator()
