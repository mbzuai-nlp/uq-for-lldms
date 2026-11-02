import numpy as np
import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class GreedyDebertaSimilarityMatrixLastBlockTraceCalculator(StatCalculator):
    """
    Calculates the DeBERTa semantic similarity between greedy sequence and sampled sequences
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
                "greedy_deberta_entail_last_block_forward_trace",
                "greedy_deberta_entail_last_block_backward_trace",
                "greedy_deberta_entail_last_block_trace",
                "greedy_deberta_contra_last_block_forward_trace",
                "greedy_deberta_contra_last_block_backward_trace",
                "greedy_deberta_contra_last_block_trace",
            ],
            ["last_block_generation_trace"],
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
        Calculates the DeBERTa semantic similarity for the last block only.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'last_block_generation_trace' (List[List[str]]): several sampling generations
                    from the last block for each input text in the batch.
            texts (List[str]): Input texts batch used for model generation.
            model (WhiteboxModel): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_deberta_entail_last_block_forward_trace' (np.ndarray): entailment probabilities
                    for pairs (greedy_last_block, sample_last_block).
                - 'greedy_deberta_entail_last_block_backward_trace' (np.ndarray): entailment probabilities
                    for pairs (sample_last_block, greedy_last_block).
                - 'greedy_deberta_entail_last_block_trace' (np.ndarray): average entailment probabilities.
                - 'greedy_deberta_contra_last_block_forward_trace' (np.ndarray): contradiction probabilities
                    for pairs (greedy_last_block, sample_last_block).
                - 'greedy_deberta_contra_last_block_backward_trace' (np.ndarray): contradiction probabilities
                    for pairs (sample_last_block, greedy_last_block).
                - 'greedy_deberta_contra_last_block_trace' (np.ndarray): average contradiction probabilities.
        """

        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size
        batch_texts = dependencies["last_block_generation_trace"]

        batch_pairs = []
        batch_pairs_b = []
        batch_invs = []
        for texts in batch_texts:
            # The last element of the trace is the greedy text's last block
            greedy_last_block = texts[-1] if texts else ""

            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_last_block], unique_texts)))
            batch_pairs_b.append(list(itertools.product(unique_texts, [greedy_last_block])))
            batch_invs.append(inv)

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        tokenizer = deberta.deberta_tokenizer

        entail_arrays_f = []
        entail_arrays_b = []
        entail_arrays = []
        contra_arrays_f = []
        contra_arrays_b = []
        contra_arrays = []

        with torch.no_grad():
            for i, pairs in enumerate(batch_pairs):
                pairs_b = batch_pairs_b[i]
                inv = batch_invs[i]

                # Handle empty pairs - add zero similarity to maintain batch size
                if len(pairs) == 0:
                    entail_arrays_f.append(np.array([0.0]))
                    entail_arrays_b.append(np.array([0.0]))
                    entail_arrays.append(np.array([0.0]))
                    contra_arrays_f.append(np.array([0.0]))
                    contra_arrays_b.append(np.array([0.0]))
                    contra_arrays.append(np.array([0.0]))
                    continue

                # Process forward pairs (greedy_last_block, sample_last_block)
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
                del logits
                torch.cuda.empty_cache()

                entail_probs_f = probs_f[:, ent_id]
                contra_probs_f = probs_f[:, contra_id]

                # Process backward pairs (sample_last_block, greedy_last_block)
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
                del logits
                torch.cuda.empty_cache()

                entail_probs_b = probs_b[:, ent_id]
                contra_probs_b = probs_b[:, contra_id]

                # Convert to numpy and apply inverse indices
                entail_arrays_f.append(entail_probs_f.cpu().numpy()[inv])
                entail_arrays_b.append(entail_probs_b.cpu().numpy()[inv])
                entail_arrays.append(
                    (entail_probs_f.cpu().numpy()[inv] + entail_probs_b.cpu().numpy()[inv]) / 2
                )

                contra_arrays_f.append(contra_probs_f.cpu().numpy()[inv])
                contra_arrays_b.append(contra_probs_b.cpu().numpy()[inv])
                contra_arrays.append(
                    (contra_probs_f.cpu().numpy()[inv] + contra_probs_b.cpu().numpy()[inv]) / 2
                )

        if not entail_arrays_f:
            return {
                "greedy_deberta_entail_last_block_forward_trace": np.array([]),
                "greedy_deberta_entail_last_block_backward_trace": np.array([]),
                "greedy_deberta_entail_last_block_trace": np.array([]),
                "greedy_deberta_contra_last_block_forward_trace": np.array([]),
                "greedy_deberta_contra_last_block_backward_trace": np.array([]),
                "greedy_deberta_contra_last_block_trace": np.array([]),
            }

        entail_arrays_f = np.stack(entail_arrays_f)
        entail_arrays_b = np.stack(entail_arrays_b)
        entail_arrays = np.stack(entail_arrays)

        contra_arrays_f = np.stack(contra_arrays_f)
        contra_arrays_b = np.stack(contra_arrays_b)
        contra_arrays = np.stack(contra_arrays)

        return {
            "greedy_deberta_entail_last_block_forward_trace": entail_arrays_f,
            "greedy_deberta_entail_last_block_backward_trace": entail_arrays_b,
            "greedy_deberta_entail_last_block_trace": entail_arrays,
            "greedy_deberta_contra_last_block_forward_trace": contra_arrays_f,
            "greedy_deberta_contra_last_block_backward_trace": contra_arrays_b,
            "greedy_deberta_contra_last_block_trace": contra_arrays,
        }


def load_calculator(cfg: Dict) -> StatCalculator:
    """
    Factory function to create a GreedyDebertaSimilarityMatrixLastBlockTraceCalculator.
    Note: This is a placeholder - the actual builder pattern uses load_stat_calculator.
    """
    # This factory function is provided for consistency but the actual instantiation
    # should use the builder pattern to share the NLI model
    raise NotImplementedError(
        "Use load_stat_calculator from greedy_deberta_similarity_last_block_trace_builder.py instead"
    )
