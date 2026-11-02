# DiffusionGemma generation mechanics & instrumentation (design agent 2, source-verified)

Model `google/diffusiongemma-26B-A4B-it` — REAL, Google DeepMind, released 2026-06-10, Apache-2.0.
25.2B total / 3.8B active MoE, encoder-decoder, block/canvas text diffusion. (Post training cutoff.)

## Verified classes / API (HF docs: transformers/model_doc/diffusion_gemma)
- `DiffusionGemmaForBlockDiffusion`, `AutoProcessor`, `DiffusionGemmaGenerationMixin/Config/Output`,
  `TextDiffusionStreamer`, `DiffusionGemma{Encoder,Decoder}Model`.
- Sampler = **`EntropyBoundSampler` + `EntropyBoundSamplerConfig`** (NOT `EntropyBoundScheduler`).
  Stopping = `StableAndConfidentStoppingCriteria`. Temp = `LinearTemperatureScheduleLogitsProcessor`.
- `generate()` → `DiffusionGemmaGenerationOutput{sequences, tokens_per_forward (NFE metric),
  past_key_values}`; **`logits=None, scores=None, hidden_states=None`** (BC stubs).
- **No `use_cache` arg** (always KV cache). `top_k` unsupported. No `trust_remote_code`.
- generation_config.json: `eos_token_id=[1,106,50]`, `pad=0`, `max_denoising_steps=48`,
  `max_new_tokens=256`, `entropy_bound=0.1`, `stability_threshold=1`, `t_max=0.8`, `t_min=0.4`,
  `confidence_threshold=0.005`. `canvas_length=256`, `vocab=262144`, 30 layers, 128 experts (8+1).
- **Uniform corruption, NO persistent mask token**: non-accepted canvas positions → random vocab
  tokens each step; acceptance is **non-monotonic** (accepted token can be demoted/re-noised).
- Two-level loop: OUTER autoregressive over canvases (finalized canvas→encoder→KV), INNER diffusion
  denoising over one canvas; self-conditioning (step logits feed next step).
- Env pitfall: import fails `torchvision::nms does not exist` (torch/torchvision mismatch) → works on
  **transformers 5.11.0**, install from source. Single-GPU needs **>60 GB (H100)** per docs (internal
  said 51.7 GB / 2×A100-40GB).
- 256 MiB/step full-canvas logits → store reductions only.

## Instrumentation (read-only, algorithm-preserving)
Two hooks; neither changes the official algorithm. **Golden rule: processor returns `scores` UNCHANGED**
(same tensor feeds acceptance + self-conditioning).

```python
class UQTraceProcessor:  # register FIRST in logits_processor → RAW pre-temperature logits
    def __init__(self): self.entropy=[]; self.maxlp=[]; self.top2gap=[]; self.argmax=[]; self.nfe=0
    def __call__(self, input_ids, scores, cur_step=None, **kw):   # scores (B, C=256, V=262144)
        self.nfe += 1
        lp = scores.log_softmax(-1); p = lp.exp()
        self.entropy.append((-(p*lp).sum(-1)).to(torch.float16).cpu())   # (B,C)
        self.maxlp.append(lp.max(-1).values.to(torch.float16).cpu())     # (B,C) log P(argmax)
        t2 = p.topk(2,-1).values
        self.top2gap.append((t2[...,0]-t2[...,1]).to(torch.float16).cpu())
        self.argmax.append(scores.argmax(-1).to(torch.int32).cpu())      # (B,C) draft
        return scores                                                    # IDENTITY

class UQTraceStreamer:
    _takes_logits = False
    def __init__(self, tok): self.tok=tok; self.drafts=[]
    def put_draft(self, value=None, logits=None, **kw):   # value=argmax_canvas (B,256) ids
        self.drafts.append(self.tok.batch_decode(value, skip_special_tokens=True))
    def put(self, value, **kw): pass
    def end(self, **kw): pass
```
Wire: `model.generate(**inputs, logits_processor=LogitsProcessorList([proc]), streamer=strm)`.
Stack per-step reductions → `(S,B,C)`; pack to `(B, num_canvases, max_steps, canvas_length)` to mirror
the repo's LLaDA `gen_trace_{entropy,logits,argmax,top2gap}` schema.

**Commit detection (argmax-stability, NOT accepted-mask — acceptance non-monotonic):**
`final[p]` from `sequences` (strip pad=0, truncate at first EOS∈{1,106,50}); align to canvas/pos.
`commit_step(p) = 1 + max{s: AM[s,p]!=final[p]}` (0 if always matches);
`confidence_at_commit(p) = maxlp[commit_step(p), p]`.
Analogs: `renoise_fraction_at_commit` (≈ mask_fraction_at_commit), `num_flips[p]` (≈ remask_count).

## Failure modes
1. `scores=None`/`logits=None` → build trace from processor; synthesize a `scores`-like tuple from
   commit-step reductions, or run ONE extra decoder_forward on the finalized canvas for final logits.
2. Never pass `use_cache`. 3. Multi-GPU sharding → keep reductions `.cpu()`. 4. Use
   `AutoProcessor.apply_chat_template(...,tokenize=True,return_dict=True,add_generation_prompt=True)`.
5. EOS is a LIST [1,106,50]; strip pad=0, truncate earliest EOS before metrics. 6. torchvision::nms →
   install transformers from source. 7. User processors run first = raw logits (re-apply temp schedule
   t=0.4→0.8 for sampler-view). 8. Don't treat accepted_mask/first-accept as commit. 9. Handle both
   `put_draft(value=)` and `(logits=)`.

## ASSUMED — verify at runtime (smoke test)
Exact hook signature (`lp(input_ids, scores, cur_step=)`, scores `(B,256,V)`, once/step, user-first);
`put_draft` arg (value ids vs logits); whether `EntropyBoundSampler.accepted_token_mask` reachable;
`diffusers==0.39.0` pin (verified floor transformers 5.11); identity-return byte-identical vs no-proc;
`out.scores is None` + `tokens_per_forward` present. Re-read generation_config.json from the loaded
model (don't hardcode). No existing DiffusionGemma code in repo/branches.
