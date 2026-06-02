from __future__ import annotations

import os
from typing import List, Sequence

import torch


INS_EMB_SINGLE_INSTRUCTIONS: List[str] = [
    "Answer with the option's letter from the given choices directly.",
    "Answer the question using a single word or phrase.",
    "What is happening in the presented picture?\nPlease describe it in one complete sentence.",
    "What is the object in the image?\nAnswer the question using a single word or phrase.",
    "Answer the question using a single word or phrase.",
    "Answer the question using a single word or phrase.",
    "What is the background of the image?\nAnswer the question using a single word or phrase.",
]


def _is_local_transformers_model_dir(model_name: str) -> bool:
    p = os.path.abspath(os.path.expanduser(model_name.strip()))
    return bool(os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")))


def _load_pretrained_or_raise(model_name: str, *, local_files_only: bool):
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as e:
        hint = (
            "Failed to load Sentence-Transformer / MiniLM (often no access to huggingface.co). Options: "
            "(1) Download sentence-transformers/all-MiniLM-L6-v2 on a machine with network and set "
            "smolora_sentence_transformer_model to that local snapshot directory (must contain config.json); "
            "(2) Use HF mirror via HF_ENDPOINT; "
            "(3) Set ins_emb_path to a pre-built ins_emb_single.pkl; "
            "(4) Set smolora_builtin_sentence_ins_emb=False and use CLIP text_tower. "
            f"Original error: {type(e).__name__}: {e}"
        )
        raise RuntimeError(hint) from e
    return tokenizer, model


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def compute_instruction_embeddings(
    sentences: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_only = _is_local_transformers_model_dir(model_name)
    if local_only:
        model_name = os.path.abspath(os.path.expanduser(model_name.strip()))
    tokenizer, model = _load_pretrained_or_raise(model_name, local_files_only=local_only)
    model.eval()
    model.to(device)
    encoded = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**encoded)
    emb = mean_pooling(out, encoded["attention_mask"])
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb.cpu().float()


def compute_default_ins_emb_matrix(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device | None = None,
) -> torch.Tensor:
    return compute_instruction_embeddings(INS_EMB_SINGLE_INSTRUCTIONS, model_name=model_name, device=device)
