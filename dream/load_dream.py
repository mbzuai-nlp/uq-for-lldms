from transformers import AutoTokenizer, AutoModel
import torch


def load_model(model_path: str, device_map: str):
    model = AutoModel.from_pretrained(
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
