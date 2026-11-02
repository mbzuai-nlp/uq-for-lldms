from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
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
    end_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if end_token_id != tokenizer.unk_token_id:
        tokenizer.eos_token_id = end_token_id

    return tokenizer
