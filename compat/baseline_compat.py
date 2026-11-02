"""Dream-7B-only runtime monkey-patches.

All patches here are installed **only** when the Dream loader
(`llada/load_dream_cached.py`) builds a Dream model/tokenizer, via
`install_model_patches` / `install_tokenizer_patches`. Nothing in this module
runs at import time and nothing is installed for other models (LLaDA-1.5 /
LLaDA-MoE / LLaDA-2.1), so non-Dream runs are unaffected.

Patches:
1. _patch_dream_attn_mask_dtype — Fast-dLLM's DreamSdpaAttention forwards the
   tokenizer's int64 attention_mask into scaled_dot_product_attention, which
   under torch>=2.5 raises "Expected attn_mask dtype to be bool or float".
   Applied to the Dream model handle.
2. _patch_dream_tokenizer_oov_safe — make argmax-driven decode robust to token
   IDs outside the tokenizer vocab (Dream's lm_head is padded past the real
   BPE vocab). Applied to the Dream tokenizer handle.
3. _patch_focus_trust_remote_code — lm_polygraph's Focus IDF tokenizer load
   calls AutoTokenizer.from_pretrained without trust_remote_code=True; Dream
   needs custom code, so the call would hit an interactive prompt under
   non-tty. Installed alongside the tokenizer patch so it is only active for
   Dream runs.
"""

import logging

log = logging.getLogger(__name__)


def _patch_dream_attn_mask_dtype(model):
    """Wrap a Dream model's forward so int attention_mask is cast to bool."""
    import torch

    orig_forward = model.forward

    def _coerce(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (
            torch.long,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.int16,
        ):
            return tensor.bool()
        return tensor

    def wrapped_forward(*args, **kwargs):
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = _coerce(kwargs["attention_mask"])
        return orig_forward(*args, **kwargs)

    model.forward = wrapped_forward
    log.info("[compat] Dream attention_mask dtype coercion installed on %s", type(model).__name__)


def _patch_focus_trust_remote_code():
    """Monkey-patch lm_polygraph Focus IDF tokenizer load to trust remote code."""
    try:
        from lm_polygraph.estimators import focus as _focus
    except ImportError:
        return
    if getattr(_focus, "_trust_remote_code_patched", False):
        return

    from transformers import AutoTokenizer

    _orig_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_from_pretrained(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        tok = _orig_from_pretrained(*args, **kwargs)
        # Dream's custom tokenizer (trust_remote_code) doesn't expose `.vocab`,
        # only `.get_vocab()`. Focus's calcu_idf does `len(tokenizer.vocab)`.
        # Inject a `vocab` attribute that mirrors get_vocab().
        if not hasattr(tok, "vocab") and hasattr(tok, "get_vocab"):
            try:
                tok.vocab = tok.get_vocab()
            except Exception as e:
                log.warning("[compat] could not seed tokenizer.vocab: %s", e)
        return tok

    _focus.AutoTokenizer.from_pretrained = _patched_from_pretrained
    _focus._trust_remote_code_patched = True
    log.info(
        "[compat] Focus AutoTokenizer.from_pretrained now defaults trust_remote_code=True "
        "and seeds .vocab from get_vocab() for custom tokenizers"
    )


def _patch_dream_tokenizer_oov_safe(tokenizer):
    """Make Dream's _convert_id_to_token return '' (not None) for OOV ids.

    Dream's lm_head emits 152064 logits for hardware-efficient sizing, but the
    actual BPE vocab only fills ~152023 of those. When argmax(-1) lands on a
    padding id, _convert_id_to_token returns None (via dict.get with no
    default), and convert_tokens_to_string explodes at "".join(tokens) with
    'expected str instance, NoneType found'.

    Replace the lookup so missing ids decode to empty string. Idempotent.
    """
    if getattr(tokenizer, "_oov_safe_patched", False):
        return
    if not hasattr(tokenizer, "decoder"):
        return

    orig = tokenizer._convert_id_to_token

    def safe_convert(index):
        tok = orig(index)
        if tok is None:
            return ""
        return tok

    tokenizer._convert_id_to_token = safe_convert
    tokenizer._oov_safe_patched = True
    log.info("[compat] Dream tokenizer _convert_id_to_token now OOV-safe (returns '' for unmapped ids)")


def install_model_patches(model):
    """Install patches that need a model handle (call after Dream model load)."""
    _patch_dream_attn_mask_dtype(model)


def install_tokenizer_patches(tokenizer):
    """Install patches that need a tokenizer handle (call after Dream tokenizer load).

    Also installs the Focus trust-remote-code patch here (rather than globally)
    so it is only active for Dream runs — the Focus IDF tokenizer is loaded
    lazily during estimation, but patching at Dream-tokenizer-load time keeps
    the change scoped to Dream and out of every other model's path.
    """
    _patch_dream_tokenizer_oov_safe(tokenizer)
    _patch_focus_trust_remote_code()
