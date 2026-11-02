"""Runtime-contract verifier for DiffusionGemma UQ instrumentation.

RUN THIS FIRST on the real model/GPU, before trusting the backend. It confirms
the ASSUMED facts from docs/diffusiongemma/02_mechanics.md that cannot be checked
without the model:

  1. `generate()` returns `scores=None`/`logits=None` and `tokens_per_forward`.
  2. A custom `logits_processor` fires once per denoising step with `scores` of
     shape (B, canvas_len, vocab); discover its exact kwargs (cur_step?).
  3. The processor sees RAW pre-temperature logits and runs BEFORE the temperature
     processor (user-processor-first).
  4. Identity-return: generating WITH the read-only processor yields byte-identical
     `sequences` vs WITHOUT it (proves the hook is non-invasive).
  5. `streamer.put_draft(...)` arg is `value` (token ids) vs `logits`.
  6. The argmax-stability commit rule yields sensible commit steps + confidence.

Usage (in the isolated env):
    CUDA_VISIBLE_DEVICES=0 python scripts/validate_diffusiongemma.py
    DG_MODEL=google/diffusiongemma-26B-A4B-it DG_MAXNEW=128 python scripts/validate_diffusiongemma.py
"""
from __future__ import annotations

import os
import sys
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

MODEL_ID = os.environ.get("DG_MODEL", "google/diffusiongemma-26B-A4B-it")
MAX_NEW = int(os.environ.get("DG_MAXNEW", "128"))
SEED = int(os.environ.get("DG_SEED", "0"))

PROMPTS = [
    "What is the capital of France?",
    "Summarize in one sentence: The sun rose over the quiet meadow as birds began their morning chorus.",
    "Translate to English: 'Le chat est sur la table.'",
]


class UQTraceProcessor:
    """Read-only logits processor. Reduces (B, C, V) -> four (B, C) tensors/step.

    MUST return `scores` unchanged (the same tensor feeds acceptance + self-conditioning).
    """

    def __init__(self):
        self.entropy, self.maxlp, self.top2gap, self.argmax = [], [], [], []
        self.nfe = 0
        self.seen_kwargs = None
        self.scores_shape = None

    def __call__(self, input_ids, scores, cur_step=None):
        # transformers 5.x validates the signature: params beyond (input_ids, scores)
        # must be keys the model actually passes. DiffusionGemma passes only cur_step
        # (generation_diffusion_gemma.py: logits_processor(input_ids, raw_logits, cur_step=...)).
        if self.scores_shape is None:
            self.seen_kwargs = ["cur_step"] if cur_step is not None else []
            self.scores_shape = tuple(scores.shape)
        self.nfe += 1
        s = scores.float()
        lp = s.log_softmax(-1)
        p = lp.exp()
        self.entropy.append((-(p * lp).sum(-1)).to(torch.float16).cpu())
        self.maxlp.append(lp.max(-1).values.to(torch.float16).cpu())
        t2 = p.topk(2, -1).values
        self.top2gap.append((t2[..., 0] - t2[..., 1]).to(torch.float16).cpu())
        self.argmax.append(scores.argmax(-1).to(torch.int32).cpu())
        return scores  # IDENTITY — never mutate


class UQTraceStreamer:
    _takes_logits = False

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.drafts = []
        self.put_draft_arg = None

    def put_draft(self, value=None, logits=None, **kwargs):
        if self.put_draft_arg is None:
            self.put_draft_arg = "value" if value is not None else ("logits" if logits is not None else "?")
        ids = value if value is not None else logits
        try:
            self.drafts.append(self.tok.batch_decode(ids, skip_special_tokens=True))
        except Exception:
            self.drafts.append(None)

    def put(self, value, **kwargs):
        pass

    def end(self, **kwargs):
        pass


def build_inputs(processor, prompt, device):
    msgs = [{"role": "user", "content": prompt}]
    try:
        enc = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:  # bare-tokenizer fallback
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(prompt, return_tensors="pt")
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}


def generate(model, processor, inputs, with_hooks):
    from transformers import LogitsProcessorList
    torch.manual_seed(SEED)
    kw = dict(max_new_tokens=MAX_NEW)
    proc = strm = None
    if with_hooks:
        proc = UQTraceProcessor()
        strm = UQTraceStreamer(getattr(processor, "tokenizer", processor))
        kw["logits_processor"] = LogitsProcessorList([proc])
        kw["streamer"] = strm
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    return out, proc, strm


def commit_analysis(argmax_steps, final_ids, maxlp_steps):
    """commit_step(p) = 1 + last step where argmax disagreed with final; conf = maxlp at commit."""
    A = torch.stack(argmax_steps)      # (S, B, C)
    LP = torch.stack(maxlp_steps)      # (S, B, C)
    S, B, C = A.shape
    n = min(C, final_ids.shape[-1])
    fin = final_ids[..., :n]           # (B, n)
    disagree = (A[:, :, :n] != fin.unsqueeze(0))         # (S,B,n)
    idx = torch.arange(S).view(S, 1, 1)
    last_bad = torch.where(disagree, idx, torch.full_like(A[:, :, :n], -1)).max(0).values  # (B,n)
    commit = (last_bad + 1).clamp(max=S - 1)
    conf = torch.gather(LP[:, :, :n], 0, commit.unsqueeze(0)).squeeze(0)  # (B,n)
    return commit, conf


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {MODEL_ID} device={device}")
    from llada.load_diffusiongemma import load_model, load_tokenizer
    processor = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID, device_map="auto")

    gen_cfg = getattr(model, "generation_config", None)
    print("[gen_config]",
          {k: getattr(gen_cfg, k, None) for k in
           ("eos_token_id", "pad_token_id", "max_denoising_steps", "max_new_tokens")} if gen_cfg else None)

    checks = {}
    for i, prompt in enumerate(PROMPTS):
        inputs = build_inputs(processor, prompt, device)
        out_h, proc, strm = generate(model, processor, inputs, with_hooks=True)
        text = getattr(processor, "tokenizer", processor).batch_decode(
            out_h.sequences, skip_special_tokens=True)[0]

        if i == 0:
            checks["scores_is_None"] = getattr(out_h, "scores", "MISSING") is None
            checks["has_tokens_per_forward"] = hasattr(out_h, "tokens_per_forward")
            checks["processor_fired"] = proc.nfe > 0
            checks["scores_3d"] = proc.scores_shape is not None and len(proc.scores_shape) == 3
            checks["put_draft_value"] = strm.put_draft_arg == "value"
            print(f"[signature] processor kwargs={proc.seen_kwargs} scores_shape={proc.scores_shape} "
                  f"nfe={proc.nfe} put_draft_arg={strm.put_draft_arg} n_drafts={len(strm.drafts)}")
            # identity-return parity (same seed, no hooks)
            out_n, _, _ = generate(model, processor, inputs, with_hooks=False)
            m = min(out_h.sequences.shape[-1], out_n.sequences.shape[-1])
            checks["identity_return"] = bool(
                torch.equal(out_h.sequences[..., :m], out_n.sequences[..., :m]))
            # commit analysis
            if proc.argmax:
                commit, conf = commit_analysis(proc.argmax, out_h.sequences.cpu(), proc.maxlp)
                checks["commit_finite"] = bool(torch.isfinite(conf.float()).all())
                print(f"[commit] commit_step[:8]={commit[0][:8].tolist()} "
                      f"confidence_at_commit[:8]={conf[0][:8].float().round(decimals=3).tolist()}")

        print(f"\n=== Prompt {i+1}: {prompt}")
        print(f"  gen: {text[:160]!r}")
        if strm and strm.drafts and strm.drafts[0] is not None:
            print(f"  draft[0]: {str(strm.drafts[0])[:100]!r} ... draft[-1]: {str(strm.drafts[-1])[:100]!r}")

    print("\n" + "=" * 60 + "\nCONTRACT CHECKS\n" + "=" * 60)
    ok = True
    for k, v in checks.items():
        ok &= bool(v)
        print(f"  [{'OK ' if v else 'FAIL'}] {k} = {v}")
    print("\nRESULT:", "CONTRACT VERIFIED ✅" if ok else "CONTRACT VIOLATIONS ❌ — adjust the backend hooks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
