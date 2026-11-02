import numpy as np
import logging
from typing import Dict, Literal
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.common import compute_sim_score
from scipy.linalg import eigh

log = logging.getLogger(__name__)


class EccentricityTrace(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Eccentricity" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    Method calculates a frobenious (euclidian) norm between all eigenvectors that are informative embeddings
    of graph Laplacian (lower norm -> closer embeddings -> higher eigenvectors -> greater uncertainty).

    This is a trace-based version that uses generation_trace instead of sample_texts.
    """

    def __init__(
        self,
        similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
        affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
        verbose: bool = False,
        thres: float = 0.9,
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
        self.thres = thres

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"EccentricityTrace_{self.similarity_score}_{self.affinity}"
        return f"EccentricityTrace_{self.similarity_score}"

    def U_Eccentricity(self, i, stats):
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

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)

        if self.thres is not None:
            keep_mask = eigenvalues < self.thres
            eigenvalues, smallest_eigenvectors = (
                eigenvalues[keep_mask],
                eigenvectors[:, keep_mask],
            )

        smallest_eigenvectors = smallest_eigenvectors.T

        C_Ecc_s_j = (-1) * np.asarray(
            [np.linalg.norm(x - x.mean(0), 2) for x in smallest_eigenvectors]
        )
        U_Ecc = np.linalg.norm(C_Ecc_s_j, 2)

        return U_Ecc, C_Ecc_s_j

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'generation_trace',
                * matrix with semantic similarities in 'semantic_matrix_entail_trace'/'semantic_matrix_contra_trace'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["generation_trace"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            res.append(self.U_Eccentricity(i, stats)[0])
        return np.array(res)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `EccentricityTrace` estimator from a config dictionary.
    """
    similarity_score = cfg.get("similarity_score", "NLI_score")
    affinity = cfg.get("affinity", "entail")
    thres = cfg.get("thres", 0.9)
    return EccentricityTrace(
        similarity_score=similarity_score, affinity=affinity, thres=thres
    )
