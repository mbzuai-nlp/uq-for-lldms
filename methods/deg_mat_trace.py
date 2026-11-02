import numpy as np
import logging
from typing import Dict, Literal
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.common import compute_sim_score

log = logging.getLogger(__name__)


class DegMatTrace(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "The Degree Matrix" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    Elements on diagonal of matrix D are sums of similarities between the particular number
    (position in matrix) and other answers. Thus, it is an average pairwise distance
    (lower values indicated smaller distance between answers which means greater uncertainty).
    """

    def __init__(
        self,
        similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
        affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
        verbose: bool = False,
    ):
        """
        Parameters:
            similarity_score (str): The argument to be processed. Possible values are:
                - 'NLI_score': As a similarity score NLI score is used.
                - 'Jaccard_score': As a similarity Jaccard score between responces is used.
            affinity (str): The argument to be processed. Possible values are. Relevant for the case of NLI similarity score:
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

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"DegMatTrace_{self.similarity_score}_{self.affinity}"
        return f"DegMatTrace_{self.similarity_score}"

    def U_DegMat(self, i, stats):
        # The Degree Matrix
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
        return np.trace(len(answers) - D) / (len(answers) ** 2)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * matrix with semantic similarities in 'semantic_matrix_entail'/'semantic_matrix_contra'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["generation_trace"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            res.append(self.U_DegMat(i, stats))
        return np.array(res)


def load_estimator(cfg: Dict) -> Estimator:
    """
    Factory function to create a `DegMatTrace` estimator from a config dictionary.
    """
    similarity_score = cfg.get("similarity_score", "NLI_score")
    affinity = cfg.get("affinity", "entail")
    return DegMatTrace(similarity_score=similarity_score, affinity=affinity)
