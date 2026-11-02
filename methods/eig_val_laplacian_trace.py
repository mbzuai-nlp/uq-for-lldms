import numpy as np
import logging
from typing import Dict, Literal
from scipy.linalg import eigh
from lm_polygraph.estimators.common import compute_sim_score
from lm_polygraph.estimators.estimator import Estimator

log = logging.getLogger(__name__)


class EigValLaplacianTrace(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Sum of Eigenvalues of the Graph Laplacian" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    A continuous analogue to the number of semantic sets (higher values means greater uncertainty).

    This is a trace-based version that uses generation_trace instead of sample_texts.
    """

    def __init__(
        self,
        similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
        affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
        verbose: bool = False,
    ):
        """
        See parameters descriptions in https://arxiv.org/abs/2305.19187.
        Parameters:
            similarity_score (str): similarity score for matrix construction. Possible values:
                - 'NLI_score': Natural Language Inference similarity
                - 'Jaccard_score': Jaccard score similarity
            affinity (str): affinity method, relevant only when similarity_score='NLI_score'. Possible values:
                - 'entail': similarity(response_1, response_2) = p_entail(response_1, response_2)
                - 'contra': similarity(response_1, response_2) = 1 - p_contra(response_1, response_2)
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        if similarity_score == "NLI_score":
            if affinity == "entail":
                super().__init__(
                    ["semantic_matrix_entail_trace", "generation_trace"], "sequence"
                )
            else:
                super().__init__(
                    ["semantic_matrix_contra_trace", "generation_trace"], "sequence"
                )
        else:
            super().__init__(["generation_trace"], "sequence")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"EigValLaplacianTrace_{self.similarity_score}_{self.affinity}"
        return f"EigValLaplacianTrace_{self.similarity_score}"

    def U_EigVal_Laplacian(self, i, stats):
        answers = stats["generation_trace"][i]

        if self.similarity_score == "NLI_score":
            if self.affinity == "entail":
                W = stats["semantic_matrix_entail_trace"][i, :, :]
            else:
                W = 1 - stats["semantic_matrix_contra_trace"][i, :, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(
                answers=answers,
                affinity=self.affinity,
                similarity_score=self.similarity_score,
            )

        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        return sum([max(0, 1 - lambda_k) for lambda_k in eigh(L, eigvals_only=True)])

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'generation_trace',
                * matrix with semantic similarities in 'semantic_matrix_entail_trace'/'semantic_matrix_contra_trace'
        """
        res = []
        for i, answers in enumerate(stats["generation_trace"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            res.append(self.U_EigVal_Laplacian(i, stats))
        return np.array(res)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `EigValLaplacianTrace` estimator from a config dictionary.
    """
    similarity_score = cfg.get("similarity_score", "NLI_score")
    affinity = cfg.get("affinity", "entail")
    return EigValLaplacianTrace(similarity_score=similarity_score, affinity=affinity)
