from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

from lm_polygraph.estimators.estimator import Estimator


TRACEDET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        epsilon: float = 1.0,
        reduction: str = "mean",
        weight=None,
    ):
        # Adapted from TraceDet utils/predictors/loss.py: Poly1CrossEntropyLoss.__init__.
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        # Adapted from TraceDet utils/predictors/loss.py: Poly1CrossEntropyLoss.forward.
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
            device=logits.device, dtype=logits.dtype
        )
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        ce = F.cross_entropy(
            input=logits, target=labels, reduction="none", weight=self.weight
        )
        poly1 = ce + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class L2Loss(nn.Module):
    def __init__(self):
        # Adapted from TraceDet utils/predictors/loss.py: L2Loss.__init__.
        super().__init__()

    def forward(self, attn):
        # Adapted from TraceDet utils/predictors/loss.py: L2Loss.forward.
        return torch.sum(torch.pow(attn, 2)).sqrt()


class ConnectLoss(nn.Module):
    def __init__(self):
        # Adapted from TraceDet utils/predictors/loss.py: ConnectLoss.__init__.
        super().__init__()

    def forward(self, logits):
        # Adapted from TraceDet utils/predictors/loss.py: ConnectLoss.forward.
        shift1 = logits[:, 1:, :]
        shift2 = logits[:, :-1, :]
        return torch.sum((shift1 - shift2).norm(p=2)) / shift1.flatten().shape[0]


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        # Adapted from TraceDet models/encoders/positional_enc.py: PositionalEncodingTF.__init__.
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        # Adapted from TraceDet models/encoders/positional_enc.py: PositionalEncodingTF.getPE.
        P_time = P_time.float()
        timescales = self.max_len ** torch.linspace(
            0, 1, self._num_timescales
        ).to(P_time.device)
        times = P_time.unsqueeze(2)
        scaled_time = times / timescales[None, None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)

    def forward(self, P_time):
        # Adapted from TraceDet models/encoders/positional_enc.py: PositionalEncodingTF.forward.
        return self.getPE(P_time)


class TransformerMVTS(nn.Module):
    def __init__(
        self,
        d_inp,
        max_len,
        n_classes,
        nhead=1,
        trans_dim_feedforward=8,
        trans_dropout=0.2,
        nlayers=3,
        aggreg="last_token",
        MAX=10000,
        d_pe=8,
    ):
        # Adapted from TraceDet models/encoders/transformer_simple.py: TransformerMVTS.__init__.
        super().__init__()
        self.d_inp = d_inp
        self.max_len = max_len
        self.n_classes = n_classes
        self.aggreg = aggreg
        self.d_pe = d_pe
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_pe + d_inp,
            nhead=nhead,
            dim_feedforward=trans_dim_feedforward,
            dropout=trans_dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.MLP_encoder = nn.Linear(d_inp, d_inp)
        d_fi = d_inp + d_pe
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.PReLU(),
            nn.Linear(d_fi, d_fi),
            nn.PReLU(),
            nn.Linear(d_fi, d_fi),
            nn.PReLU(),
            nn.Linear(d_fi, n_classes),
        )
        nn.init.xavier_uniform_(self.MLP_encoder.weight)

    def embed(self, src, times, aggregate=True, src_mask=None):
        # Adapted from TraceDet models/encoders/transformer_simple.py: TransformerMVTS.embed.
        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)
        if src_mask is None and torch.any(times < -1e5):
            src_mask = (times < -1e5).transpose(0, 1)
        lengths = torch.sum(times > 0, dim=0)
        src = self.MLP_encoder(src)
        pe = self.pos_encoder(times).to(src.device)
        x = torch.cat([pe, src], axis=2)
        output_preagg = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        if not aggregate:
            return output_preagg
        if self.aggreg == "mean":
            lengths2 = lengths.unsqueeze(1)
            if src_mask is not None:
                output = torch.sum(
                    output_preagg
                    * (
                        1
                        - src_mask.transpose(0, 1)
                        .unsqueeze(-1)
                        .repeat(1, 1, output_preagg.shape[-1])
                        .float()
                    ),
                    dim=0,
                ) / (lengths2 + 1)
            else:
                output = torch.sum(output_preagg, dim=0) / (lengths2 + 1)
        else:
            if src_mask is not None:
                lengths = (1 - src_mask).sum(dim=1).clamp(min=1)
                idx = (
                    (lengths - 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(1, 1, output_preagg.shape[-1])
                )
                output = output_preagg.gather(0, idx).squeeze(0)
            else:
                output = output_preagg[-1, :, :]
        return output

    def forward(self, src, times, get_embedding=False, get_agg_embed=False):
        # Adapted from TraceDet models/encoders/transformer_simple.py: TransformerMVTS.forward.
        out_full = self.embed(src, times, aggregate=False)
        out = self.embed(src, times, aggregate=True)
        out = nn.LayerNorm(out.size(-1)).to(out.device)(out)
        output = self.mlp(out)
        if get_embedding:
            return output, out_full
        if get_agg_embed:
            return output, out, out_full
        return output


class MaskGenerator(nn.Module):
    def __init__(
        self,
        d_z,
        max_len,
        d_pe=8,
        trans_dim_feedforward=32,
        nhead=1,
        n_dec_layers=2,
        tau=1.0,
        use_ste=True,
    ):
        # Adapted from TraceDet models/TimeHalu.py: MaskGenerator.__init__.
        super().__init__()
        self.d_z = d_z
        self.d_pe = d_pe
        self.max_len = max_len
        self.tau = tau
        self.use_ste = use_ste
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_z,
            nhead=nhead,
            dim_feedforward=trans_dim_feedforward,
            batch_first=False,
        )
        self.mask_decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)
        self.time_prob_net = nn.Linear(d_z, 2)
        self.ln_in_tgt = nn.LayerNorm(d_z)
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len)
        self.init_weights()

    def init_weights(self):
        # Adapted from TraceDet models/TimeHalu.py: MaskGenerator.init_weights.
        def iweights(m):
            # Adapted from TraceDet models/TimeHalu.py: MaskGenerator.init_weights local iweights.
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.time_prob_net.apply(iweights)

    def reparameterize(self, total_mask):
        # Adapted from TraceDet models/TimeHalu.py: MaskGenerator.reparameterize.
        total_mask_prob = total_mask.softmax(dim=-1)
        return F.gumbel_softmax(
            torch.log(total_mask_prob + 1e-9), tau=self.tau, hard=self.use_ste
        )[..., 1]

    def forward(self, z_seq, src, times):
        # Adapted from TraceDet models/TimeHalu.py: MaskGenerator.forward.
        x = torch.cat([src, self.pos_encoder(times).to(src.device)], dim=-1)
        if torch.any(times < -1e5):
            tgt_mask = (times < -1e5).transpose(0, 1)
        else:
            tgt_mask = None
        x = self.ln_in_tgt(x)
        z_seq_dec = self.mask_decoder(
            tgt=x, memory=z_seq, tgt_key_padding_mask=tgt_mask
        )
        p_time = self.time_prob_net(z_seq_dec)
        total_mask_reparameterize = self.reparameterize(p_time.transpose(0, 1))
        total_mask = p_time.transpose(0, 1).softmax(dim=-1)[..., 1].unsqueeze(-1)
        return total_mask, total_mask_reparameterize


class TimeHalu(nn.Module):
    def __init__(
        self,
        clf,
        extractor,
        loss_weight_dict,
        gsat_r,
        d_inp=128,
        d_pe=8,
    ):
        # Adapted from TraceDet models/TimeHalu.py: TimeHalu.__init__.
        super().__init__()
        d_fi = d_inp + d_pe
        self.d_inp = d_inp
        self.clf = clf
        self.extractor = extractor
        self.loss_weight_dict = loss_weight_dict
        self.L2_loss_fn = L2Loss()
        self.connected_loss = ConnectLoss()
        self.gsat_r = gsat_r
        self.clf_criterion = Poly1CrossEntropyLoss(
            num_classes=2, epsilon=1.0, weight=None, reduction="mean"
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.LayerNorm(d_fi),
            nn.Linear(d_fi, 2),
        )

    def __loss__(self, mask_logits, y_pred_masked, y_pred, Y):
        # Adapted from TraceDet models/TimeHalu.py: TimeHalu.__loss__.
        masked_pred_loss = self.clf_criterion(y_pred_masked, Y)
        orig_pred_loss = self.clf_criterion(y_pred, Y)
        mask_loss = self.loss_weight_dict["gsat"] * self.L2_loss_fn(
            mask_logits
        ) + self.loss_weight_dict["connect"] * self.connected_loss(mask_logits)
        loss = 0.2 * orig_pred_loss + masked_pred_loss + mask_loss
        return loss, masked_pred_loss, mask_loss

    def forward(self, X, times, Y, captum_input=False):
        # Adapted from TraceDet models/TimeHalu.py: TimeHalu.forward entropy branch.
        emb = self.clf.embed(X, times, aggregate=False)
        mask_prob, rep_mask = self.extractor(emb, X, times)
        ste_mask_rs = mask_prob.transpose(0, 1)
        emb_masked = emb * ste_mask_rs
        emb_masked = torch.sum(emb_masked, dim=0) / (
            torch.sum(ste_mask_rs, dim=0) + 1e-8
        )
        emb = torch.sum(emb, dim=0) / self.d_inp
        emb = nn.LayerNorm(emb.size(-1)).to(emb.device)(emb)
        emb_masked = nn.LayerNorm(emb_masked.size(-1)).to(emb_masked.device)(
            emb_masked
        )
        y_pred = self.mlp(emb)
        y_pred_masked = self.mlp(emb_masked)
        loss, pred_loss, mask_loss = self.__loss__(
            mask_prob, y_pred_masked, y_pred, Y
        )
        return {
            "mask_logits": mask_prob,
            "mask_reparam": rep_mask,
            "loss": loss,
            "pred_loss": pred_loss,
            "mask_loss": mask_loss,
            "y_pred_masked": y_pred_masked,
            "y_pred": y_pred,
        }


class TraceDetTimeHalu(Estimator):
    def __init__(
        self,
        num_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 2e-4,
        gsat: float = 0.1,
        gsat_r: float = 0.1,
        connect: float = 0.0,
        nlayers: int = 3,
        dropout: float = 0.2,
        d_pe: int = 8,
        metric_thr: float = 0.3,
        device: str = None,
    ):
        # Adapted from TraceDet train_main.py: TimeHalu training arguments.
        super().__init__(
            ["train_gen_trace_entropy", "gen_trace_entropy", "train_metrics"],
            "sequence",
        )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gsat = gsat
        self.gsat_r = gsat_r
        self.connect = connect
        self.nlayers = nlayers
        self.dropout = dropout
        self.d_pe = d_pe
        self.metric_thr = metric_thr
        self.device = torch.device(device) if device is not None else TRACEDET_DEVICE
        self.is_fitted = False
        self.model = None
        self.max_len = None
        self.d_inp = None

    def __str__(self):
        # Adapted from TraceDet train_main.py: save path names use the TimeHalu method name.
        return "TraceDetTimeHalu_entropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Adapted from TraceDet train_main.py: train TimeHalu first, then evaluate y_pred_masked.
        if not self.is_fitted:
            train_x, train_times = self._prepare_trace(stats["train_gen_trace_entropy"])
            train_y = self._make_target(stats["train_metrics"])
            self._fit(train_x, train_times, train_y)
        x, times = self._prepare_trace(stats["gen_trace_entropy"])
        return self._predict(x, times)

    def _prepare_trace(self, trace) -> Tuple[np.ndarray, np.ndarray]:
        # Adapted from TraceDet DLLM_generate/data_generator.py: transposes saved trace tensors to time-major training data.
        arr = self._to_numpy(trace).astype(np.float32)
        if arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], arr.shape[3])
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D, 3D or 4D trace tensor, got {arr.shape}")

        samples = []
        for sample in arr:
            valid = np.isfinite(sample).any(axis=-1)
            sample = sample[valid]
            if len(sample) == 0:
                sample = np.zeros((1, arr.shape[-1]), dtype=np.float32)
            sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
            samples.append(sample)

        max_len = self.max_len or max(len(sample) for sample in samples)
        d_inp = self.d_inp or arr.shape[-1]
        x = np.zeros((len(samples), max_len, d_inp), dtype=np.float32)
        times = np.full((len(samples), max_len), -1e6, dtype=np.float32)
        for i, sample in enumerate(samples):
            sample = self._align_features(sample, d_inp)
            n_steps = min(len(sample), max_len)
            x[i, :n_steps] = sample[:n_steps]
            times[i, :n_steps] = np.arange(1, n_steps + 1)
        return x, times

    def _to_numpy(self, value):
        # Adapted from TraceDet DLLM_generate/data_generator.py: torch.save tensors are loaded back for training.
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                value = value.float()
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _align_features(self, sample, d_inp):
        # Adapted from TraceDet train_main.py: TimeHalu assumes a fixed d_inp feature dimension.
        if sample.shape[-1] == d_inp:
            return sample
        if sample.shape[-1] > d_inp:
            return sample[:, :d_inp]
        pad = np.zeros((sample.shape[0], d_inp - sample.shape[-1]), dtype=sample.dtype)
        return np.concatenate([sample, pad], axis=-1)

    def _make_target(self, train_metrics):
        # Adapted from lm-polygraph supervised_methods saplma.py: supervised UE learns 1 - target metric.
        target = self._to_numpy(train_metrics).astype(np.float32).reshape(-1)
        unique_vals = np.unique(target)
        if np.all((unique_vals == 0) | (unique_vals == 1)):
            return (1 - target).astype(np.int64)
        return (target < self.metric_thr).astype(np.int64)

    def _fit(self, train_x, train_times, train_y):
        # Adapted from TraceDet trainers/train_TimeHalu.py: epoch loop over TimeHalu batches.
        set_seed(42)
        self.max_len = train_x.shape[1]
        self.d_inp = train_x.shape[2]
        clf = TransformerMVTS(
            d_inp=self.d_inp,
            max_len=self.max_len,
            n_classes=2,
            nlayers=self.nlayers,
            trans_dim_feedforward=8,
            trans_dropout=self.dropout,
            d_pe=self.d_pe,
            aggreg="last_token",
        )
        extractor = MaskGenerator(
            d_z=self.d_inp + self.d_pe,
            d_pe=self.d_pe,
            max_len=self.max_len,
            tau=1.0,
            use_ste=True,
        )
        self.model = TimeHalu(
            clf,
            extractor,
            {"gsat": self.gsat, "connect": self.connect},
            gsat_r=self.gsat_r,
            d_inp=self.d_inp,
            d_pe=self.d_pe,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1)
        x_t = torch.tensor(train_x, dtype=torch.float32)
        times_t = torch.tensor(train_times, dtype=torch.float32)
        y_t = torch.tensor(train_y, dtype=torch.long)
        for _ in range(self.num_epochs):
            self.model.train()
            permutation = torch.randperm(len(x_t))
            for start in range(0, len(x_t), self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                x_batch = x_t[batch_idx].transpose(0, 1).to(self.device)
                times_batch = times_t[batch_idx].transpose(0, 1).to(self.device)
                y_batch = y_t[batch_idx].to(self.device)
                out = self.model(x_batch, times_batch, y_batch)
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def _predict(self, x, times):
        # Adapted from TraceDet train_main.py: uses y_pred_masked and AUROC scores from class 1.
        x_t = torch.tensor(x, dtype=torch.float32)
        times_t = torch.tensor(times, dtype=torch.float32)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x_t), self.batch_size):
                x_batch = x_t[start : start + self.batch_size].transpose(0, 1).to(self.device)
                times_batch = times_t[start : start + self.batch_size].transpose(0, 1).to(self.device)
                dummy_y = torch.zeros(x_batch.shape[1], dtype=torch.long, device=self.device)
                out = self.model(x_batch, times_batch, dummy_y)
                prob = F.softmax(out["y_pred_masked"], dim=-1)[:, 1]
                preds.append(prob.detach().cpu().numpy())
        return np.concatenate(preds)


def load_estimator(cfg: Dict) -> Estimator:
    # Adapted from TraceDet train_main.py: constructs TimeHalu from CLI-like config values.
    return TraceDetTimeHalu(
        num_epochs=cfg.get("num_epochs", 100),
        batch_size=cfg.get("batch_size", 64),
        lr=cfg.get("lr", 2e-4),
        gsat=cfg.get("gsat", 0.1),
        gsat_r=cfg.get("gsat_r", 0.1),
        connect=cfg.get("connect", 0.0),
        nlayers=cfg.get("nlayers", 3),
        dropout=cfg.get("dropout", 0.2),
        d_pe=cfg.get("d_pe", 8),
        metric_thr=cfg.get("metric_thr", 0.3),
        device=cfg.get("device", None),
    )
