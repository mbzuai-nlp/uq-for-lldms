import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from qwen3.modeling_qwen3_refusion import Qwen3ForCausalLM
from transformers import AutoTokenizer
import torch


def _assert_cuda_usable(device_map):
    if device_map in (None, "cpu"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for ReFusion loading.")
    errors = []
    for idx in range(torch.cuda.device_count()):
        try:
            torch.empty(1, device=f"cuda:{idx}")
            return
        except Exception as exc:
            errors.append(f"cuda:{idx}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No usable CUDA devices for ReFusion loading. " + "; ".join(errors))


def load_model(model_path: str, device_map: str):
    _assert_cuda_usable(device_map)
    if device_map == "custom_4gpu":
        max_memory = {
            0: "1GiB",
            1: "5GiB",
            2: "9GiB",
            3: "9GiB",
        }
        model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            use_safetensors=True,
        )
    else:
        model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_safetensors=True,
        )
    model.eval()

    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = model.config.n_layers
    if not hasattr(model.config, "num_attention_heads"):
        model.config.num_attention_heads = model.config.n_heads

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
    if tokenizer.eos_token_id is None:
        raise ValueError("ReFusion tokenizer does not define an EOS token.")

    return tokenizer
