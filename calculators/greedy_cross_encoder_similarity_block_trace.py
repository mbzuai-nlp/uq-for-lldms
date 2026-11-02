import numpy as np

import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity between greedy sequence and sampled sequences
    for EACH BLOCK separately.

    Uses block_generation_traces which has structure [bs][block][step].
    For each batch item and each block, compares the greedy text (last step) with all steps.
    Returns similarity matrices.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return (
            [
                "greedy_sentence_similarity_block_forward",
                "greedy_sentence_similarity_block_backward",
                "greedy_sentence_similarity_block",
            ],
            ["block_generation_traces"],
        )

    def __init__(
        self,
        batch_size: int = 10,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-large",
    ):
        super().__init__()
        self.crossencoder_setup = False
        self.batch_size = batch_size
        self.cross_encoder_name = cross_encoder_name

    def _setup(self, device="cuda"):
        self.crossencoder = CrossEncoder(self.cross_encoder_name, device=device)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        device = model.device()

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        batch_block_traces = dependencies["block_generation_traces"]
        # Structure: [bs][block][step]

        # For each batch item, collect all blocks
        batch_blocks = []
        for block_traces in batch_block_traces:
            blocks_for_batch = []
            for block_trace in block_traces:
                if block_trace:  # Non-empty block
                    blocks_for_batch.append(block_trace)
            batch_blocks.append(blocks_for_batch)

        # Process each block for each batch item
        all_sim_f = []
        all_sim_b = []
        all_sim = []

        for blocks_for_batch in batch_blocks:
            bs_sim_f = []
            bs_sim_b = []
            bs_sim = []

            for block_trace in blocks_for_batch:
                # Get greedy text (last step)
                greedy_text = block_trace[-1] if block_trace else ""

                # Get unique texts and their inverse mapping
                unique_texts, inv = np.unique(block_trace, return_inverse=True)

                if len(unique_texts) == 0:
                    # Empty block - add zero similarity
                    bs_sim_f.append(np.array([0.0]))
                    bs_sim_b.append(np.array([0.0]))
                    bs_sim.append(np.array([0.0]))
                    continue

                # Create pairs: (greedy_text, each_unique_text)
                pairs = list(itertools.product([greedy_text], unique_texts))
                pairs_b = [(b, a) for a, b in pairs]

                # Predict similarities
                sim_scores_f = self.crossencoder.predict(pairs, batch_size=self.batch_size)
                sim_scores_b = self.crossencoder.predict(pairs_b, batch_size=self.batch_size)
                sim_scores = (sim_scores_f + sim_scores_b) / 2

                # Restore original order using inverse mapping
                bs_sim_f.append(sim_scores_f[inv])
                bs_sim_b.append(sim_scores_b[inv])
                bs_sim.append(sim_scores[inv])

            # Keep as list of arrays (blocks may have different lengths)
            if bs_sim_f:
                all_sim_f.append(bs_sim_f)
                all_sim_b.append(bs_sim_b)
                all_sim.append(bs_sim)
            else:
                all_sim_f.append([])
                all_sim_b.append([])
                all_sim.append([])

        return {
            "greedy_sentence_similarity_block_forward": all_sim_f,
            "greedy_sentence_similarity_block_backward": all_sim_b,
            "greedy_sentence_similarity_block": all_sim,
        }


def load_calculator(cfg: Dict) -> StatCalculator:
    """
    Factory function to create a GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator.
    """
    batch_size = cfg.get("batch_size", 10)
    cross_encoder_name = cfg.get("cross_encoder_name", "cross-encoder/stsb-roberta-large")
    return GreedyCrossEncoderSimilarityMatrixBlockTraceCalculator(
        batch_size=batch_size,
        cross_encoder_name=cross_encoder_name,
    )
