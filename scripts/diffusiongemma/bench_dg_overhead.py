#!/usr/bin/env python
"""Measure the UQ overhead of the DiffusionGemma backend.

Reviewer weakness (3) says runtime results are hardware-specific. The hardware-independent
quantity is the RELATIVE overhead of collecting the uncertainty signals on top of the
generation the model performs anyway. Measured here as wall-clock with the read-only trace
hooks attached vs. the same generation with no hooks.

Reports median over repeats to blunt scheduler noise, after a warm-up pass.
"""
import sys, time, statistics
sys.path.insert(0, "/home/artem.shelmanov/vlad/uncertainty_dllm_dg")
import torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
from whitebox_diffusiongemma import WhiteboxModelDiffusionGemma

MODEL = "google/diffusiongemma-26B-A4B-it"
REPEATS = 7
MAX_NEW = 128
PROMPTS = [
    "Summarize the following conversation in one or two sentences.\n"
    "Anna: are we still on for tomorrow?\nBen: yes, 6pm at the usual place\n"
    "Anna: perfect, I'll book a table\nBen: great, see you then",
    "Question: Which river flows through the city of Vienna?\nAnswer:",
    "Translate the following German text into English, give only the translation.\n"
    "Original: Die Konferenz wurde auf nächste Woche verschoben.\nTranslation:",
]

print("loading...", flush=True)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
proc = AutoProcessor.from_pretrained(MODEL)
tok = getattr(proc, "tokenizer", proc)
wb = WhiteboxModelDiffusionGemma(model, tok, processor=proc, model_path=MODEL, instruct=True)
print("loaded\n", flush=True)

batch = wb.tokenize(PROMPTS)
batch = {k: v.to(wb.device()) for k, v in batch.items()}


def timed(fn, label):
    fn()  # warm-up, excluded
    ts = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print(f"{label:34s} median={med:7.3f}s  min={min(ts):7.3f}s  max={max(ts):7.3f}s", flush=True)
    return med


def bare():
    """Raw model.generate: no logits processor, no streamer, no trace collection."""
    with torch.inference_mode():
        model.generate(input_ids=batch["input_ids"],
                       attention_mask=batch.get("attention_mask"),
                       max_new_tokens=MAX_NEW)


def with_uq():
    """Backend generate: trace hooks attached + stat assembly."""
    with torch.inference_mode():
        wb.generate(**batch, max_new_tokens=MAX_NEW)


t_bare = timed(bare, "generation only (no UQ)")
t_uq = timed(with_uq, "generation + UQ trace hooks")

overhead = 100.0 * (t_uq - t_bare) / t_bare
print()
print(f"UQ overhead = {overhead:+.1f}% of generation wall-clock "
      f"({t_uq - t_bare:+.3f}s on {t_bare:.3f}s)")
print(f"batch={len(PROMPTS)} prompts, max_new_tokens={MAX_NEW}, repeats={REPEATS}, bf16, "
      f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
print()
print("Note: this is the cost of the trajectory-only signals (trace entropy, trace probs,")
print("confidence-at-commit, MSP), which need no extra decoder forward passes. The")
print("semantic-consistency estimators add auxiliary NLI/cross-encoder passes on top.")
