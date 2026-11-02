"""Load Dream 7B using vendored Fast-dLLM model code with KV-cache support.

Fast-dLLM's DreamModel has cache-aware attention layers that support
both prefix cache (concatenation) and dual cache (replace_position).
The standard HF Dream model does NOT support use_cache.
"""
import os

import torch
from dream.model.modeling_dream import DreamModel
from transformers import AutoTokenizer
from transformers.trainer_utils import load_sharded_checkpoint
from transformers.utils.hub import cached_file


def load_model(model_path: str, device_map: str):
    # Avoid passing device_map=... to from_pretrained: it triggers accelerate's
    # add_hook_to_module path, which on torch 2.6+cu124 (CIAI) wraps
    # model.forward as a functools.partial whose post_forward filters
    # MaskedLMOutputWithPastKeyValues.logits to None — causing the
    # generate.py:481 'NoneType' is not subscriptable crash. Dream-7B fits on
    # a single 48 GB GPU, so we just load on CPU then .to(device).
    model = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Transformers 5's dynamic loader reports all keys as loaded for this
    # legacy custom model but leaves its parameters initialized. Load the same
    # sharded safetensors explicitly through torch.nn.Module.load_state_dict.
    index_file = cached_file(model_path, "model.safetensors.index.json")
    load_sharded_checkpoint(
        model,
        os.path.dirname(index_file),
        strict=True,
        prefer_safe=True,
    )
    # device_map="auto" comes from polygraph_eval; tensor.to("auto") is not
    # valid. Treat any non-tensor-target value as "cuda" since we now load
    # everything to a single GPU rather than dispatch.
    target_device = device_map if device_map in ("cuda", "cpu") or (isinstance(device_map, str) and device_map.startswith(("cuda:", "cpu:"))) else "cuda"
    model.to(target_device)
    # Transformers 5 may materialize non-persistent RoPE buffers from the meta
    # device without their initialized values. Recompute them on the target
    # device before the first forward pass.
    model.reset_rope_parameters()
    model.eval()

    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = model.config.n_layers
    if not hasattr(model.config, "num_attention_heads"):
        model.config.num_attention_heads = model.config.n_heads

    # Coerce int64 attention_mask -> bool before SDPA in DreamSdpaAttention
    # (required for torch>=2.5 + lm_polygraph baseline calculators that pass
    # tokenizer attention_mask directly).
    from compat.baseline_compat import install_model_patches
    install_model_patches(model)

    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Make _convert_id_to_token return '' instead of None for unmapped ids in
    # the lm_head's hardware-padded vocab range, so argmax outputs can be
    # decoded without TypeError in batch_decode -> convert_tokens_to_string.
    from compat.baseline_compat import install_tokenizer_patches
    install_tokenizer_patches(tokenizer)

    return tokenizer
