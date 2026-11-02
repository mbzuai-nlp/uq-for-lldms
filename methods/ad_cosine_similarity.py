import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from lm_polygraph.estimators.estimator import Estimator


class ADCosineSimilarity(Estimator):
    """Whole-block mean-pooled hidden-state dissimilarity."""

    def __init__(self, layer=None):
        super().__init__(
            [
                "hidden_states_trace",
                "hidden_states_trace_mask",
                "hidden_states_trace_block_length",
                "greedy_tokens",
            ],
            "sequence",
        )
        self.layer = None if layer is None else int(layer)

    def __str__(self):
        if self.layer is not None:
            return f"AD-CosineSim-LastBlock-WholeBlock-Layer{self.layer}"
        return "AD-CosineSim-LastBlock-WholeBlock"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        traces = torch.as_tensor(stats["hidden_states_trace"]).float().cpu()
        masks = torch.as_tensor(stats["hidden_states_trace_mask"]).bool().cpu()
        debug = os.environ.get("AD_COSINE_DEBUG") == "1"

        if debug:
            print(
                f"[AD-COS][input] traces={tuple(traces.shape)} "
                f"masks={tuple(masks.shape)} "
                f"greedy_batch={len(stats['greedy_tokens'])}",
                flush=True,
            )

        if traces.ndim != 5 or masks.shape != traces.shape[:3]:
            raise ValueError(
                "Expected hidden_states_trace [B, N, S, X, H] and "
                "hidden_states_trace_mask [B, N, S]"
            )
        if self.layer is not None and not 0 <= self.layer < traces.shape[-2]:
            raise ValueError(
                f"Layer {self.layer} is not in [0, {traces.shape[-2]})"
            )

        scores = torch.zeros(traces.shape[0], dtype=torch.float32)
        for batch_idx in range(traces.shape[0]):
            generated_length = len(stats["greedy_tokens"][batch_idx])
            if generated_length == 0:
                continue

            block_width = int(
                stats["hidden_states_trace_block_length"][batch_idx]
            )
            block_idx = min((generated_length - 1) // block_width, traces.shape[1] - 1)
            block_states = traces[batch_idx, block_idx]
            pooled_states = (
                block_states.mean(dim=-2)
                if self.layer is None
                else block_states[:, self.layer]
            )
            step_mask = masks[batch_idx, block_idx]
            valid_states = pooled_states[step_mask]

            if debug:
                print(
                    f"[AD-COS][batch={batch_idx}] generated_length={generated_length} "
                    f"block_width={block_width} block_idx={block_idx} "
                    f"block_states={tuple(block_states.shape)} "
                    f"pooled_states={tuple(pooled_states.shape)} "
                    f"step_mask={tuple(step_mask.shape)} valid_steps={int(step_mask.sum())} "
                    f"valid_states={tuple(valid_states.shape)}",
                    flush=True,
                )
                for step_idx, (raw_state, pooled_state, is_valid) in enumerate(
                    zip(block_states, pooled_states, step_mask)
                ):
                    print(
                        f"[AD-COS][batch={batch_idx}][step={step_idx}] "
                        f"raw={tuple(raw_state.shape)} pooled={tuple(pooled_state.shape)} "
                        f"valid={bool(is_valid)}",
                        flush=True,
                    )
            if valid_states.shape[0] < 3:
                continue

            trajectory_states = valid_states[1:]
            final_state = trajectory_states[-1].unsqueeze(0)
            prev_states = trajectory_states[:-1]
            similarities = F.cosine_similarity(
                prev_states,
                final_state,
                dim=-1,
                eps=1e-6,
            ).clamp(-1.0, 1.0)
            dissimilarities = 1.0 - similarities
            score = dissimilarities.mean()
            scores[batch_idx] = score

            if debug:
                print(
                    f"[AD-COS][batch={batch_idx}][output] "
                    f"prev_states={tuple(prev_states.shape)} "
                    f"final_state={tuple(final_state.shape)} "
                    f"similarities={tuple(similarities.shape)} "
                    f"cos={similarities.tolist()} "
                    f"one_minus_cos={dissimilarities.tolist()} "
                    f"score={tuple(score.shape)} value={score.item():.8f}",
                    flush=True,
                )

        return scores.numpy()


def load_estimator(cfg: Dict) -> Estimator:
    return ADCosineSimilarity(layer=cfg.get("layer"))
