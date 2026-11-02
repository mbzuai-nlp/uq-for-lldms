import numpy as np

import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class GreedyCrossEncoderSimilarityMatrixLastBlockTraceCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity between greedy sequence and sampled sequences
    using ONLY THE LAST BLOCK of the generation trace.

    This is designed for datasets like GSM8K where the final answer is in the last block.
    Instead of comparing full texts (which include reasoning steps that may vary),
    this compares only the last block where the final answer appears.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return (
            [
                "greedy_sentence_similarity_last_block_forward_trace",
                "greedy_sentence_similarity_last_block_backward_trace",
                "greedy_sentence_similarity_last_block_trace",
            ],
            ["last_block_generation_trace"],
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

        batch_texts = dependencies["last_block_generation_trace"]

        batch_pairs = []
        batch_invs = []
        for texts in batch_texts:
            # The last element of the trace is the greedy text's last block
            greedy_last_block = texts[-1] if texts else ""

            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_last_block], unique_texts)))
            batch_invs.append(inv)

        sim_arrays_f = []
        sim_arrays_b = []
        sim_arrays = []
        for i, pairs in tqdm(enumerate(batch_pairs)):
            # Handle empty pairs - add zero similarity to maintain batch size
            if len(pairs) == 0:
                # Add single zero similarity for this batch element
                sim_arrays_f.append(np.array([0.0]))
                sim_arrays_b.append(np.array([0.0]))
                sim_arrays.append(np.array([0.0]))
                continue

            pairs_b = [(b, a) for a, b in pairs]
            sim_scores_f = self.crossencoder.predict(pairs, batch_size=self.batch_size)
            sim_scores_b = self.crossencoder.predict(
                pairs_b, batch_size=self.batch_size
            )

            inv = batch_invs[i]

            sim_arrays_f.append(sim_scores_f[inv])
            sim_arrays_b.append(sim_scores_b[inv])
            sim_arrays.append((sim_scores_f[inv] + sim_scores_b[inv]) / 2)

        sim_arrays_f = np.stack(sim_arrays_f)
        sim_arrays_b = np.stack(sim_arrays_b)
        sim_arrays = np.stack(sim_arrays)

        return {
            "greedy_sentence_similarity_last_block_forward_trace": sim_arrays_f,
            "greedy_sentence_similarity_last_block_backward_trace": sim_arrays_b,
            "greedy_sentence_similarity_last_block_trace": sim_arrays,
        }


def load_calculator(cfg: Dict) -> StatCalculator:
    """
    Factory function to create a GreedyCrossEncoderSimilarityMatrixLastBlockTraceCalculator.
    """
    batch_size = cfg.get("batch_size", 10)
    cross_encoder_name = cfg.get("cross_encoder_name", "cross-encoder/stsb-roberta-large")
    return GreedyCrossEncoderSimilarityMatrixLastBlockTraceCalculator(
        batch_size=batch_size,
        cross_encoder_name=cross_encoder_name,
    )
