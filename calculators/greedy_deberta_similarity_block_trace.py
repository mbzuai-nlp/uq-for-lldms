import numpy as np
import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class GreedyDebertaSimilarityMatrixBlockTraceCalculator(StatCalculator):
    """
    Calculates the DeBERTa semantic similarity between greedy sequence and sampled sequences
    for EACH BLOCK separately.

    Uses block_generation_traces which has structure [bs][block][step].
    For each batch item and each block, compares the greedy text (last step) with all steps.
    Returns entailment and contradiction probabilities.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return (
            [
                "greedy_deberta_entail_block_forward",
                "greedy_deberta_entail_block_backward",
                "greedy_deberta_entail_block",
                "greedy_deberta_contra_block_forward",
                "greedy_deberta_contra_block_backward",
                "greedy_deberta_contra_block",
            ],
            ["block_generation_traces"],
        )

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the DeBERTa semantic similarity for each block.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'block_generation_traces' (List[List[List[str]]]): per-block generation traces
                    with structure [bs][block][step].
            texts (List[str]): Input texts batch used for model generation.
            model (WhiteboxModel): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_deberta_entail_block_forward' (List[List[np.ndarray]]): entailment probabilities
                    for pairs (greedy_text, sample_text) for each block.
                - 'greedy_deberta_entail_block_backward' (List[List[np.ndarray]]): entailment probabilities
                    for pairs (sample_text, greedy_text) for each block.
                - 'greedy_deberta_entail_block' (List[List[np.ndarray]]): average entailment probabilities.
                - 'greedy_deberta_contra_block_forward' (List[List[np.ndarray]]): contradiction probabilities
                    for pairs (greedy_text, sample_text) for each block.
                - 'greedy_deberta_contra_block_backward' (List[List[np.ndarray]]): contradiction probabilities
                    for pairs (sample_text, greedy_text) for each block.
                - 'greedy_deberta_contra_block' (List[List[np.ndarray]]): average contradiction probabilities.
        """

        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size
        batch_block_traces = dependencies["block_generation_traces"]
        # Structure: [bs][block][step]

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        tokenizer = deberta.deberta_tokenizer

        # Process each batch item
        all_entail_f = []
        all_entail_b = []
        all_entail = []
        all_contra_f = []
        all_contra_b = []
        all_contra = []

        for block_traces in batch_block_traces:
            bs_entail_f = []
            bs_entail_b = []
            bs_entail = []
            bs_contra_f = []
            bs_contra_b = []
            bs_contra = []

            for block_trace in block_traces:
                if not block_trace:  # Empty block
                    bs_entail_f.append(np.array([0.0]))
                    bs_entail_b.append(np.array([0.0]))
                    bs_entail.append(np.array([0.0]))
                    bs_contra_f.append(np.array([0.0]))
                    bs_contra_b.append(np.array([0.0]))
                    bs_contra.append(np.array([0.0]))
                    continue

                # Get greedy text (last step)
                greedy_text = block_trace[-1]

                # Get unique texts and their inverse mapping
                unique_texts, inv = np.unique(block_trace, return_inverse=True)

                # Create pairs: (greedy_text, each_unique_text)
                pairs = list(itertools.product([greedy_text], unique_texts))
                pairs_b = list(itertools.product(unique_texts, [greedy_text]))

                with torch.no_grad():
                    # Process forward pairs (greedy_text, sample_text)
                    dl_f = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
                    probs_f = []
                    for first_texts, second_texts in dl_f:
                        encoded = tokenizer(
                            list(first_texts),
                            list(second_texts),
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        logits = deberta.deberta(**encoded).logits
                        probs_f.append(softmax(logits))

                    probs_f = torch.cat(probs_f, dim=0)
                    del encoded, logits
                    torch.cuda.empty_cache()

                    entail_probs_f = probs_f[:, ent_id]
                    contra_probs_f = probs_f[:, contra_id]

                    # Process backward pairs (sample_text, greedy_text)
                    dl_b = torch.utils.data.DataLoader(pairs_b, batch_size=deberta_batch_size)
                    probs_b = []
                    for first_texts, second_texts in dl_b:
                        encoded = tokenizer(
                            list(first_texts),
                            list(second_texts),
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        logits = deberta.deberta(**encoded).logits
                        probs_b.append(softmax(logits))

                    probs_b = torch.cat(probs_b, dim=0)
                    del encoded, logits
                    torch.cuda.empty_cache()

                    entail_probs_b = probs_b[:, ent_id]
                    contra_probs_b = probs_b[:, contra_id]

                    # Convert to numpy and apply inverse indices
                    entail_f = entail_probs_f.cpu().numpy()[inv]
                    entail_b = entail_probs_b.cpu().numpy()[inv]
                    entail_avg = (entail_f + entail_b) / 2

                    contra_f = contra_probs_f.cpu().numpy()[inv]
                    contra_b = contra_probs_b.cpu().numpy()[inv]
                    contra_avg = (contra_f + contra_b) / 2

                    bs_entail_f.append(entail_f)
                    bs_entail_b.append(entail_b)
                    bs_entail.append(entail_avg)
                    bs_contra_f.append(contra_f)
                    bs_contra_b.append(contra_b)
                    bs_contra.append(contra_avg)

            # Keep as list of arrays (blocks may have different lengths)
            all_entail_f.append(bs_entail_f)
            all_entail_b.append(bs_entail_b)
            all_entail.append(bs_entail)
            all_contra_f.append(bs_contra_f)
            all_contra_b.append(bs_contra_b)
            all_contra.append(bs_contra)

        return {
            "greedy_deberta_entail_block_forward": all_entail_f,
            "greedy_deberta_entail_block_backward": all_entail_b,
            "greedy_deberta_entail_block": all_entail,
            "greedy_deberta_contra_block_forward": all_contra_f,
            "greedy_deberta_contra_block_backward": all_contra_b,
            "greedy_deberta_contra_block": all_contra,
        }


def load_calculator(cfg: Dict) -> StatCalculator:
    """
    Factory function to create a GreedyDebertaSimilarityMatrixBlockTraceCalculator.
    Note: This is a placeholder - the actual builder pattern uses load_stat_calculator.
    """
    # This factory function is provided for consistency but the actual instantiation
    # should use the builder pattern to share the NLI model
    raise NotImplementedError(
        "Use load_stat_calculator from greedy_deberta_similarity_block_trace_builder.py instead"
    )
