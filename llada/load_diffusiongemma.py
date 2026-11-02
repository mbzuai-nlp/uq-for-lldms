"""Load Google DiffusionGemma (`google/diffusiongemma-26B-A4B-it`).

DiffusionGemma is an encoder-decoder BLOCK/canvas text-diffusion LM with UNIFORM
corruption (no persistent mask token) — architecturally different from LLaDA/Dream.
It is served by `transformers>=5.11` via `DiffusionGemmaForBlockDiffusion` and a
multimodal `AutoProcessor`. `generate()` returns `scores=None` and does NOT accept
`use_cache` (the KV cache is always on). See docs/diffusiongemma/02_mechanics.md.

This module only loads weights + processor; the UQ generation loop + trace hooks
live in `whitebox_diffusiongemma.py`.
"""
import torch
from transformers import AutoProcessor

# Checkpoint EOS ids (from generation_config.json — re-read at runtime, do not trust blindly).
DIFFUSIONGEMMA_EOS_IDS = [1, 106, 50]


def load_model(model_path: str, device_map="auto", dtype: str = "bfloat16", **kwargs):
    """Load DiffusionGemmaForBlockDiffusion sharded across visible GPUs.

    ~51.7 GB in bf16 → needs >60 GB single-GPU (H100 / RunPod 97 GB) or
    device_map="auto" across 2-4x A100-40GB. Never pass `use_cache`.
    """
    # Imported lazily so the module can be inspected in an env without transformers 5.11+.
    from transformers import DiffusionGemmaForBlockDiffusion  # VERIFY@runtime: class name

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    model.eval()
    # Attention calculators read these from the top-level config (as LLaDA's config exposes them).
    for key in ("num_hidden_layers", "num_attention_heads", "hidden_size"):
        setattr(model.config, key, getattr(model.config.text_config, key))

    # Surface the checkpoint's own generation config (EOS ids, max_denoising_steps,
    # entropy_bound, t_min/t_max, ...). Prefer the loaded config over hardcoded values.
    gen_cfg = getattr(model, "generation_config", None)
    eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if eos is None:
        if gen_cfg is not None:
            gen_cfg.eos_token_id = DIFFUSIONGEMMA_EOS_IDS
    return model


def load_tokenizer(model_path: str, **kwargs):
    """Return the multimodal AutoProcessor (holds .tokenizer + chat template).

    The whitebox wrapper uses `processor.apply_chat_template(...)` for prompts and
    `processor.tokenizer` / `processor.batch_decode` for decoding.
    """
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return processor
