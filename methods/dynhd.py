import itertools
import logging
import string
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

from lm_polygraph.estimators.estimator import Estimator


DYNHD_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger("lm_polygraph")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
BOILERPLATE = {"answer", "question", "context", "user", "assistant", "system"}


def binary_auroc(y_true, y_score):
    # Adapted from DynHD Sec. 4.1: validation AUROC selects the detector.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_log_loss(y_true, y_score):
    # Adapted from DynHD Sec. 4.1: fallback validation score when one class is absent.
    eps = 1e-7
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), eps, 1.0 - eps)
    return -float((y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score)).mean())


class DynHDModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        # Adapted from DynHD Sec. 3.2: reference dynamics generator and temporal detector.
        super().__init__()
        self.reference = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.project = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, evidence, query_emb, step_emb, mask):
        # Adapted from DynHD Sec. 3.2: x_t = [a_t; a_hat_t; delta a_t], then temporal aggregation.
        query = query_emb.unsqueeze(1).expand(-1, evidence.shape[1], -1)
        ref = self.reference(torch.cat([query, step_emb], dim=-1))
        delta = torch.zeros_like(evidence)
        delta[:, 1:] = evidence[:, :-1] - evidence[:, 1:]
        z = self.project(torch.cat([evidence, ref, delta], dim=-1))
        attn = self.attn_w(torch.tanh(self.attn_u(z))).squeeze(-1)
        attn = attn.masked_fill(~mask, -1e4)
        weights = torch.softmax(attn, dim=-1)
        pooled = (z * weights.unsqueeze(-1)).sum(1)
        return self.classifier(pooled), ref, weights


class DynHD(Estimator):
    def __init__(
        self,
        num_epochs: int = 50,
        pretrain_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 3e-4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        top_k: int = 5,
        path_lambda: float = 0.1,
        rebound_lambda: float = 0.1,
        margin_quantile: float = 0.95,
        margin_beta: float = 0.1,
        weight_decay_stage1: float = 0.0,
        weight_decay_stage2: float = 0.0,
        warmup_ratio: float = 0.0,
        grid_search: bool = False,
        validation_fraction: float = 0.1765,
        refit_on_full: bool = True,
        lr_stage1_grid=None,
        lr_stage2_grid=None,
        weight_decay_stage1_grid=None,
        weight_decay_stage2_grid=None,
        path_lambda_grid=None,
        rebound_lambda_grid=None,
        warmup_ratio_grid=None,
        metric_thr: float = 0.3,
        freeze_reference: bool = True,
        device: str = None,
    ):
        # Adapted from DynHD Sec. 3: learns hallucination scores from denoising dynamics deviation.
        super().__init__(
            [
                "train_gen_trace_entropy",
                "train_gen_trace_argmax",
                "train_input_embeddings_mean",
                "train_gen_trace_embeddings",
                "train_metrics",
                "gen_trace_entropy",
                "gen_trace_argmax",
                "input_embeddings_mean",
                "gen_trace_embeddings",
                "tokenizer",
            ],
            "sequence",
        )
        self.num_epochs = num_epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.top_k = top_k
        self.path_lambda = path_lambda
        self.rebound_lambda = rebound_lambda
        self.margin_quantile = margin_quantile
        self.margin_beta = margin_beta
        self.weight_decay_stage1 = weight_decay_stage1
        self.weight_decay_stage2 = weight_decay_stage2
        self.warmup_ratio = warmup_ratio
        self.grid_search = grid_search
        self.validation_fraction = validation_fraction
        self.refit_on_full = refit_on_full
        self.lr_stage1_grid = self._as_grid(lr_stage1_grid, [3e-4, 1e-3])
        self.lr_stage2_grid = self._as_grid(lr_stage2_grid, [1e-4, 3e-4, 1e-3])
        self.weight_decay_stage1_grid = self._as_grid(
            weight_decay_stage1_grid, [0.0, 0.01, 0.05]
        )
        self.weight_decay_stage2_grid = self._as_grid(
            weight_decay_stage2_grid, [0.0, 1e-4, 1e-3]
        )
        self.path_lambda_grid = self._as_grid(path_lambda_grid, self._lambda_grid())
        self.rebound_lambda_grid = self._as_grid(rebound_lambda_grid, self._lambda_grid())
        self.warmup_ratio_grid = self._as_grid(warmup_ratio_grid, [0.0, 0.3])
        self.metric_thr = metric_thr
        self.freeze_reference = freeze_reference
        self.device = torch.device(device) if device is not None else DYNHD_DEVICE
        self.is_fitted = False
        self.model = None
        self.max_len = None
        self.embed_dim = None
        self.path_margin = None
        self.rebound_margin = None
        self._token_cache = {}

    def _lambda_grid(self):
        # Adapted from DynHD Table 4: regularizer weights in [0.0, 0.4] with step 0.05.
        return [round(float(value), 2) for value in np.arange(0.0, 0.4001, 0.05)]

    def _as_grid(self, value, default):
        # Adapted from DynHD Table 4: keeps Hydra list values as plain Python floats.
        if value is None:
            return default
        return [float(item) for item in list(value)]

    def __str__(self):
        # Adapted from DynHD paper title: denoising dynamics deviation learning.
        return "DynHD"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Adapted from DynHD Sec. 3.3: pretrain reference dynamics, then train detector.
        tokenizer = stats["tokenizer"]
        if not self.is_fitted:
            train = self._build_features(
                stats["train_gen_trace_entropy"],
                stats["train_gen_trace_argmax"],
                stats["train_input_embeddings_mean"],
                stats["train_gen_trace_embeddings"],
                tokenizer,
                fit=True,
            )
            train_y = self._make_target(stats["train_metrics"])
            self._fit(train, train_y)
        test = self._build_features(
            stats["gen_trace_entropy"],
            stats["gen_trace_argmax"],
            stats["input_embeddings_mean"],
            stats["gen_trace_embeddings"],
            tokenizer,
            fit=False,
        )
        return self._predict(test)

    def _build_features(self, entropy, argmax, query_emb, step_emb, tokenizer, fit):
        # Adapted from DynHD Sec. 3.1: semantic-aware evidence construction from entropy trajectories.
        entropy = self._flatten_trace(entropy).astype(np.float32)
        argmax = self._align_like(
            self._flatten_trace(argmax).astype(np.int64), entropy.shape, -1
        )
        query_emb = self._align_query_embeddings(query_emb, entropy.shape[0], fit)
        step_emb = self._align_step_embeddings(step_emb, entropy.shape[0], entropy.shape[1])

        evidence, lengths, steps = [], [], []
        for ent_i, tok_i, emb_i in zip(entropy, argmax, step_emb):
            valid_time = np.isfinite(ent_i).any(axis=-1)
            ent_i = np.nan_to_num(ent_i[valid_time], nan=0.0, posinf=0.0, neginf=0.0)
            tok_i = tok_i[valid_time]
            if len(valid_time) == len(emb_i):
                emb_i = emb_i[valid_time[: len(emb_i)]]
            else:
                emb_i = emb_i[: len(ent_i)]
            if len(ent_i) == 0:
                ent_i = np.zeros((1, entropy.shape[-1]), dtype=np.float32)
                tok_i = np.full((1, entropy.shape[-1]), -1, dtype=np.int64)
                emb_i = np.zeros(
                    (1, self.embed_dim or query_emb.shape[-1]), dtype=np.float32
                )
            evidence.append(self._entropy_evidence(ent_i, tok_i, tokenizer))
            lengths.append(len(evidence[-1]))
            steps.append(self._align_last_dim(emb_i, self.embed_dim or query_emb.shape[-1]))

        max_len = max(lengths) if fit or self.max_len is None else self.max_len
        if fit:
            self.max_len = max_len
        out_evidence = np.zeros((len(evidence), max_len, 3), dtype=np.float32)
        out_steps = np.zeros((len(evidence), max_len, query_emb.shape[-1]), dtype=np.float32)
        mask = np.zeros((len(evidence), max_len), dtype=bool)
        for i, (ev, st) in enumerate(zip(evidence, steps)):
            n = min(len(ev), max_len)
            out_evidence[i, :n] = ev[:n]
            out_steps[i, :n] = st[:n]
            mask[i, :n] = True
        return {
            "evidence": out_evidence,
            "query": query_emb,
            "step": out_steps,
            "mask": mask,
        }

    def _entropy_evidence(self, entropy, tokens, tokenizer):
        # Adapted from DynHD Sec. 3.1: [mean entropy, max entropy, top-k mean entropy].
        rows = []
        for ent_row, tok_row in zip(entropy, tokens):
            mask = np.isfinite(ent_row) & self._semantic_mask(tok_row, tokenizer)
            vals = ent_row[mask]
            if len(vals) == 0:
                rows.append([0.0, 0.0, 0.0])
                continue
            k = min(self.top_k, len(vals))
            topk = np.partition(vals, -k)[-k:]
            rows.append([float(vals.mean()), float(vals.max()), float(topk.mean())])
        return np.asarray(rows, dtype=np.float32)

    def _semantic_mask(self, tokens, tokenizer):
        # Adapted from DynHD Sec. 3.1: filters structural, punctuation, boilerplate, stopword, and subword tokens.
        return np.asarray(
            [self._is_semantic_token(int(token), tokenizer) for token in tokens],
            dtype=bool,
        )

    def _is_semantic_token(self, token_id: int, tokenizer) -> bool:
        # Adapted from DynHD Sec. 3.1: keeps only semantic evidence tokens.
        if token_id in self._token_cache:
            return self._token_cache[token_id]
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        valid = token_id >= 0 and token_id not in special_ids
        if valid:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            piece = (
                tokenizer.convert_ids_to_tokens(token_id)
                if hasattr(tokenizer, "convert_ids_to_tokens")
                else text
            )
            stripped = text.strip()
            lowered = stripped.strip(":").lower()
            punct = set(string.punctuation)
            valid = bool(stripped)
            valid = valid and lowered not in STOPWORDS and lowered not in BOILERPLATE
            valid = valid and not all(ch in punct for ch in stripped)
            valid = valid and not (stripped.startswith("<") and stripped.endswith(">"))
            valid = valid and not str(piece).startswith("##")
        self._token_cache[token_id] = valid
        return valid

    def _flatten_trace(self, value):
        # Adapted from DynHD Sec. 2 notation: stores trajectories as sample-major T x L arrays.
        arr = self._to_numpy(value)
        if arr.ndim == 4:
            return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], arr.shape[3])
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            return arr[:, :, None]
        raise ValueError(f"Expected 2D, 3D or 4D trace tensor, got {arr.shape}")

    def _to_numpy(self, value):
        # Adapted from lm-polygraph supervised methods: stat tensors are converted before training.
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                value = value.float()
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _align_like(self, value, shape, pad_value):
        # Adapted from DynHD variable-length setting here: pads missing trace entries with neutral values.
        if value.shape == shape:
            return value
        out = np.full(shape, pad_value, dtype=value.dtype)
        slices = tuple(slice(0, min(value.shape[i], shape[i])) for i in range(len(shape)))
        out[slices] = value[slices]
        return out

    def _align_query_embeddings(self, value, batch_size, fit):
        # Adapted from DynHD Sec. 3.2: query embedding conditions the reference dynamics.
        arr = self._to_numpy(value).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = self._align_batch(arr, batch_size)
        if fit:
            self.embed_dim = arr.shape[-1]
        return self._align_last_dim(arr, self.embed_dim or arr.shape[-1])

    def _align_step_embeddings(self, value, batch_size, max_time):
        # Adapted from DynHD Sec. 3.2: timestep conditioning is replaced here by mean generation-token embeddings.
        arr = self._to_numpy(value).astype(np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, None, :], max_time, axis=1)
        if arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], arr.shape[-1])
        arr = self._align_batch(arr, batch_size)
        if arr.shape[1] < max_time:
            pad = np.zeros(
                (arr.shape[0], max_time - arr.shape[1], arr.shape[-1]),
                dtype=arr.dtype,
            )
            arr = np.concatenate([arr, pad], axis=1)
        return self._align_last_dim(arr[:, :max_time], self.embed_dim or arr.shape[-1])

    def _align_batch(self, value, batch_size):
        # Adapted from lm-polygraph batch stats: keeps current batch size aligned with generated traces.
        if value.shape[0] == batch_size:
            return value
        out = np.zeros((batch_size, *value.shape[1:]), dtype=value.dtype)
        n = min(batch_size, value.shape[0])
        out[:n] = value[:n]
        return out

    def _align_last_dim(self, value, dim):
        # Adapted from DynHD compact implementation: pads or truncates embedding dimensions.
        if value.shape[-1] == dim:
            return value
        if value.shape[-1] > dim:
            return value[..., :dim]
        pad = np.zeros((*value.shape[:-1], dim - value.shape[-1]), dtype=value.dtype)
        return np.concatenate([value, pad], axis=-1)

    def _make_target(self, train_metrics):
        # Adapted from DynHD Sec. 2: y=1 denotes hallucinated/incorrect answers.
        target = self._to_numpy(train_metrics).astype(np.float32).reshape(-1)
        unique_vals = np.unique(target)
        if np.all((unique_vals == 0) | (unique_vals == 1)):
            return (1 - target).astype(np.int64)
        return (target < self.metric_thr).astype(np.int64)

    def _fit(self, train, train_y):
        # Adapted from DynHD Table 4: optional validation search over the authors' hyperparameter grid.
        tensors = self._to_tensors(train)
        y = torch.tensor(train_y, dtype=torch.long)
        if self.grid_search:
            params = self._select_hyperparameters(tensors, y)
            if self.refit_on_full:
                self._fit_with_params(tensors, y, torch.arange(len(y)), params)
        else:
            params = self._default_params()
            self._fit_with_params(tensors, y, torch.arange(len(y)), params)
        self.is_fitted = True

    def _default_params(self):
        # Adapted from DynHD Table 4: fixed config uses one point from the same search space.
        return {
            "lr_stage1": self.lr,
            "lr_stage2": self.lr,
            "weight_decay_stage1": self.weight_decay_stage1,
            "weight_decay_stage2": self.weight_decay_stage2,
            "path_lambda": self.path_lambda,
            "rebound_lambda": self.rebound_lambda,
            "warmup_ratio": self.warmup_ratio,
        }

    def _select_hyperparameters(self, tensors, y):
        # Adapted from DynHD Sec. 4.1: selects hyperparameters by validation performance.
        train_idx, val_idx = self._train_val_split(y)
        best_stage1, best_reference_state, best_loss = None, None, float("inf")
        for lr_stage1, wd_stage1 in itertools.product(
            self.lr_stage1_grid, self.weight_decay_stage1_grid
        ):
            params = self._default_params()
            params.update(
                {
                    "lr_stage1": lr_stage1,
                    "weight_decay_stage1": wd_stage1,
                }
            )
            self._fit_reference_only(tensors, y, train_idx, params)
            loss = self._reference_validation_loss(tensors, y, val_idx)
            if loss < best_loss:
                best_loss = loss
                best_stage1 = {
                    "lr_stage1": lr_stage1,
                    "weight_decay_stage1": wd_stage1,
                }
                best_reference_state = self._reference_state()

        best_params, best_score = None, -float("inf")
        stage2_grid = itertools.product(
            self.lr_stage2_grid,
            self.weight_decay_stage2_grid,
            self.path_lambda_grid,
            self.rebound_lambda_grid,
            self.warmup_ratio_grid,
        )
        for lr_stage2, wd_stage2, path_lambda, rebound_lambda, warmup_ratio in stage2_grid:
            params = {
                **self._default_params(),
                **best_stage1,
                "lr_stage2": lr_stage2,
                "weight_decay_stage2": wd_stage2,
                "path_lambda": path_lambda,
                "rebound_lambda": rebound_lambda,
                "warmup_ratio": warmup_ratio,
            }
            self._fit_with_params(
                tensors,
                y,
                train_idx,
                params,
                reference_state=best_reference_state,
            )
            score = self._validation_score(tensors, y, val_idx)
            if score > best_score:
                best_score = score
                best_params = params

        log.info(
            "DynHD best hyperparameters: %s, validation score: %.6f",
            best_params,
            best_score,
        )
        return best_params

    def _train_val_split(self, y):
        # Adapted from DynHD Sec. 4.1: keeps a validation split for hyperparameter selection.
        rng = np.random.default_rng(42)
        train_parts, val_parts = [], []
        y_np = y.cpu().numpy()
        for label in np.unique(y_np):
            idx = np.where(y_np == label)[0]
            rng.shuffle(idx)
            n_val = int(round(len(idx) * self.validation_fraction))
            if len(idx) > 1:
                n_val = min(max(n_val, 1), len(idx) - 1)
            else:
                n_val = 0
            val_parts.append(idx[:n_val])
            train_parts.append(idx[n_val:])
        val_idx = np.concatenate([part for part in val_parts if len(part)])
        train_idx = np.concatenate([part for part in train_parts if len(part)])
        if len(val_idx) == 0 or len(train_idx) == 0:
            all_idx = np.arange(len(y_np))
            rng.shuffle(all_idx)
            split = int(round(len(all_idx) * (1 - self.validation_fraction)))
            if len(all_idx) > 1:
                split = min(max(split, 1), len(all_idx) - 1)
            else:
                split = len(all_idx)
            train_idx, val_idx = all_idx[:split], all_idx[split:]
        return torch.tensor(train_idx, dtype=torch.long), torch.tensor(
            val_idx, dtype=torch.long
        )

    def _fit_with_params(self, tensors, y, train_idx, params, reference_state=None):
        # Adapted from DynHD Sec. 3.3: factual pretraining followed by detector training.
        set_seed(42)
        self.model = DynHDModel(self.embed_dim, self.hidden_dim, self.dropout).to(
            self.device
        )
        self._reset_margins()
        if reference_state is None:
            self._pretrain_reference(tensors, y, train_idx, params)
        else:
            self.model.reference.load_state_dict(reference_state)
        if self.freeze_reference:
            for param in self.model.reference.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=params["lr_stage2"],
            weight_decay=params["weight_decay_stage2"],
        )
        for epoch in range(self.num_epochs):
            permutation = train_idx[torch.randperm(len(train_idx))]
            self.model.train()
            reg_scale = self._warmup_scale(epoch, params["warmup_ratio"])
            for start in range(0, len(permutation), self.batch_size):
                idx = permutation[start : start + self.batch_size]
                batch = self._select_batch(tensors, idx)
                y_batch = y[idx].to(self.device)
                logits, ref, weights = self.model(**batch)
                loss = F.cross_entropy(logits, y_batch)
                loss = loss + self._regularization_loss(
                    batch, ref, weights, y_batch, params, reg_scale
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _fit_reference_only(self, tensors, y, train_idx, params):
        # Adapted from DynHD Sec. 3.3: evaluates stage-1 hyperparameters in isolation.
        set_seed(42)
        self.model = DynHDModel(self.embed_dim, self.hidden_dim, self.dropout).to(
            self.device
        )
        self._pretrain_reference(tensors, y, train_idx, params)

    def _reference_state(self):
        # Adapted from DynHD two-stage training: reuses the selected factual reference dynamics.
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.reference.state_dict().items()
        }

    def _reset_margins(self):
        # Adapted from DynHD Algorithm 1: initializes adaptive factual margins for stage 2.
        self.path_margin = None
        self.rebound_margin = None

    def _pretrain_reference(self, tensors, y, train_idx, params):
        # Adapted from DynHD Sec. 3.3: trains reference dynamics on factual samples.
        factual = train_idx[y[train_idx] == 0]
        if len(factual) == 0 or self.pretrain_epochs <= 0:
            return
        optimizer = torch.optim.AdamW(
            self.model.reference.parameters(),
            lr=params["lr_stage1"],
            weight_decay=params["weight_decay_stage1"],
        )
        for _ in range(self.pretrain_epochs):
            permutation = factual[torch.randperm(len(factual))]
            self.model.train()
            for start in range(0, len(permutation), self.batch_size):
                idx = permutation[start : start + self.batch_size]
                batch = self._select_batch(tensors, idx)
                _, ref, _ = self.model(**batch)
                mask = batch["mask"]
                loss = F.smooth_l1_loss(ref[mask], batch["evidence"][mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _regularization_loss(self, batch, ref, weights, y, params, reg_scale):
        # Adapted from DynHD Eq. 7-12: attention-weighted scores with EMA-margin hinge losses.
        spath, sreb = self._deviation_scores(batch, ref, weights)
        path_margin = self._update_margin("path_margin", spath, y)
        rebound_margin = self._update_margin("rebound_margin", sreb, y)
        path_loss = self._hinge_loss(spath, path_margin, y)
        rebound_loss = self._hinge_loss(sreb, rebound_margin, y)
        return reg_scale * (
            params["path_lambda"] * path_loss
            + params["rebound_lambda"] * rebound_loss
        )

    def _deviation_scores(self, batch, ref, weights):
        # Adapted from DynHD Eq. 7-8: spath and sreb use temporal weights omega_t.
        evidence = batch["evidence"]
        mask = batch["mask"].float()
        weights = weights * mask
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        spath = (weights * torch.abs(evidence - ref).sum(-1)).sum(1)
        if evidence.shape[1] < 2:
            return spath, torch.zeros_like(spath)
        rebound = torch.relu(evidence[:, :-1] - evidence[:, 1:]).pow(2).sum(-1)
        rebound_mask = (batch["mask"][:, :-1] & batch["mask"][:, 1:]).float()
        sreb = (weights[:, 1:] * rebound_mask * rebound).sum(1)
        return spath, sreb

    def _update_margin(self, name, scores, y):
        # Adapted from DynHD Eq. 9: EMA boundary from factual batch scores.
        old_margin = getattr(self, name)
        factual_scores = scores[y == 0].detach().float()
        if len(factual_scores):
            batch_margin = torch.quantile(factual_scores, self.margin_quantile)
            margin = (
                batch_margin
                if old_margin is None
                else (1 - self.margin_beta) * old_margin.to(scores.device)
                + self.margin_beta * batch_margin
            )
            setattr(self, name, margin.detach())
            return margin
        if old_margin is None:
            return torch.zeros((), device=scores.device)
        return old_margin.to(scores.device)

    def _hinge_loss(self, scores, margin, y):
        # Adapted from DynHD Eq. 10-11: (1-y) * s + y * max(0, margin - s).
        y = y.float()
        return ((1 - y) * scores + y * torch.relu(margin - scores)).mean()

    def _warmup_scale(self, epoch, warmup_ratio):
        # Adapted from DynHD Table 4: lambda warmup ratio is searched over {0, 0.3}.
        if warmup_ratio <= 0:
            return 1.0
        warmup_epochs = max(1, int(round(self.num_epochs * warmup_ratio)))
        return min(1.0, float(epoch + 1) / warmup_epochs)

    def _reference_validation_loss(self, tensors, y, val_idx):
        # Adapted from DynHD Sec. 3.3: reference dynamics are validated on factual samples.
        factual = val_idx[y[val_idx] == 0]
        idx = factual if len(factual) else val_idx
        if len(idx) == 0:
            return float("inf")
        losses = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(idx), self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                batch = self._select_batch(tensors, batch_idx)
                _, ref, _ = self.model(**batch)
                mask = batch["mask"]
                losses.append(F.smooth_l1_loss(ref[mask], batch["evidence"][mask]))
        return float(torch.stack(losses).mean().cpu())

    def _validation_score(self, tensors, y, val_idx):
        # Adapted from DynHD Sec. 4.1: validation AUROC selects the detector hyperparameters.
        if len(val_idx) == 0:
            return -float("inf")
        y_true = y[val_idx].cpu().numpy()
        probs = self._predict_tensors(tensors, val_idx)
        if len(np.unique(y_true)) == 2:
            return binary_auroc(y_true, probs)
        return -binary_log_loss(y_true, probs)

    def _to_tensors(self, data):
        # Adapted from lm-polygraph supervised estimators: numpy features become torch training tensors.
        return {
            "evidence": torch.tensor(data["evidence"], dtype=torch.float32),
            "query_emb": torch.tensor(data["query"], dtype=torch.float32),
            "step_emb": torch.tensor(data["step"], dtype=torch.float32),
            "mask": torch.tensor(data["mask"], dtype=torch.bool),
        }

    def _select_batch(self, tensors, idx):
        # Adapted from lm-polygraph supervised estimators: slices a mini-batch.
        return {key: value[idx].to(self.device) for key, value in tensors.items()}

    def _predict_tensors(self, tensors, indices):
        # Adapted from DynHD Sec. 3.2: class-1 probability is hallucination uncertainty.
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start : start + self.batch_size]
                batch = self._select_batch(tensors, idx)
                logits, _, _ = self.model(**batch)
                preds.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(preds)

    def _predict(self, data):
        # Adapted from DynHD Sec. 3.2: class-1 probability is hallucination uncertainty.
        tensors = self._to_tensors(data)
        indices = torch.arange(len(data["evidence"]), dtype=torch.long)
        return self._predict_tensors(tensors, indices)


def load_estimator(cfg: Dict) -> Estimator:
    # Adapted from lm-polygraph custom estimators: constructs DynHD from Hydra config values.
    return DynHD(
        num_epochs=cfg.get("num_epochs", 50),
        pretrain_epochs=cfg.get("pretrain_epochs", 20),
        batch_size=cfg.get("batch_size", 64),
        lr=cfg.get("lr", 3e-4),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.1),
        top_k=cfg.get("top_k", 5),
        path_lambda=cfg.get("path_lambda", 0.1),
        rebound_lambda=cfg.get("rebound_lambda", 0.1),
        margin_quantile=cfg.get("margin_quantile", 0.95),
        margin_beta=cfg.get("margin_beta", 0.1),
        weight_decay_stage1=cfg.get("weight_decay_stage1", 0.0),
        weight_decay_stage2=cfg.get("weight_decay_stage2", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        grid_search=cfg.get("grid_search", False),
        validation_fraction=cfg.get("validation_fraction", 0.1765),
        refit_on_full=cfg.get("refit_on_full", True),
        lr_stage1_grid=cfg.get("lr_stage1_grid", None),
        lr_stage2_grid=cfg.get("lr_stage2_grid", None),
        weight_decay_stage1_grid=cfg.get("weight_decay_stage1_grid", None),
        weight_decay_stage2_grid=cfg.get("weight_decay_stage2_grid", None),
        path_lambda_grid=cfg.get("path_lambda_grid", None),
        rebound_lambda_grid=cfg.get("rebound_lambda_grid", None),
        warmup_ratio_grid=cfg.get("warmup_ratio_grid", None),
        metric_thr=cfg.get("metric_thr", 0.3),
        freeze_reference=cfg.get("freeze_reference", True),
        device=cfg.get("device", None),
    )
