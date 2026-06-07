"""Backbone registry: resolve active backbone, paths, and checkpoint naming."""
from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Any

_SUPPORTED = ("llava", "internvl")
_DEFAULT = "llava"


def normalize_backbone_id(value: str | None) -> str:
    bid = (value or _DEFAULT).strip().lower()
    if bid not in _SUPPORTED:
        raise ValueError(f"Unknown backbone {bid!r}; supported: {list(_SUPPORTED)}")
    return bid


@lru_cache(maxsize=8)
def _load_backbone_module(backbone_id: str) -> ModuleType:
    bid = normalize_backbone_id(backbone_id)
    return __import__(f"config.backbone.{bid}", fromlist=["*"])


@lru_cache(maxsize=8)
def _load_paths_module(backbone_id: str) -> ModuleType:
    bid = normalize_backbone_id(backbone_id)
    return __import__(f"config.paths.{bid}_paths", fromlist=["*"])


def resolve_backbone_id(cli: str | None = None) -> str:
    """Priority: CLI > env PRISM_BACKBONE > config/run_config.BACKBONE_DEFAULT > llava."""
    if cli is not None and str(cli).strip():
        return normalize_backbone_id(cli)
    env = os.environ.get("PRISM_BACKBONE", "").strip()
    if env:
        return normalize_backbone_id(env)
    try:
        from config import run_config  # type: ignore

        return normalize_backbone_id(getattr(run_config, "BACKBONE_DEFAULT", _DEFAULT))
    except Exception:
        return _DEFAULT


def get_backbone_config(backbone_id: str | None = None) -> ModuleType:
    return _load_backbone_module(resolve_backbone_id(backbone_id))


def get_paths(backbone_id: str | None = None) -> dict[str, str]:
    bid = resolve_backbone_id(backbone_id)
    mod = _load_paths_module(bid)
    keys = (
        "BASE_MODEL_PATH",
        "VISION_TOWER_PATH",
        "ROUTING_VISION_TOWER_PATH",
        "CLIP_PATH",
        "PRETRAIN_MM_PROJECTOR",
        "CHECKPOINT_DIR",
        "RESULT_DIR",
        "DEEPSPEED_CONFIG",
    )
    out: dict[str, str] = {}
    for key in keys:
        val = getattr(mod, key, None)
        if val is not None:
            out[key] = str(val)
    out.setdefault("VISION_TOWER_PATH", out.get("CLIP_PATH", ""))
    out.setdefault("ROUTING_VISION_TOWER_PATH", out.get("CLIP_PATH", ""))
    # Legacy aliases used across the codebase
    if "CLIP_PATH" in out:
        out.setdefault("TEXT_TOWER_PATH", out["CLIP_PATH"])
    if "VISION_TOWER_PATH" in out:
        out.setdefault("MLLM_VISION_TOWER_PATH", out["VISION_TOWER_PATH"])
    return out


def get_embedded_vision_image_size(backbone_id: str | None = None) -> int:
    """Input resolution for InternViT weights bundled in InternVL-Chat (typically 336)."""
    bid = resolve_backbone_id(backbone_id)
    if bid != "internvl":
        return 224
    mod = _load_paths_module(bid)
    return int(getattr(mod, "EMBEDDED_VISION_IMAGE_SIZE", 336))


def get_backbone_display_name(backbone_id: str | None = None) -> str:
    bid = resolve_backbone_id(backbone_id)
    return {"llava": "LLaVA", "internvl": "InternVL"}.get(bid, bid.upper())


def resolve_infer_precision() -> str:
    """``config/run_config.INFER_PRECISION``: bf16 | fp16 | 4bit | 8bit."""
    try:
        from config import run_config  # type: ignore

        return str(getattr(run_config, "INFER_PRECISION", "bf16")).strip().lower()
    except Exception:
        return "bf16"


def infer_precision_flags(
    *,
    load_4bit: bool | None = None,
    load_8bit: bool | None = None,
) -> tuple[bool, bool, Any]:
    """
    Map precision to (load_8bit, load_4bit, torch_dtype).

    CLI ``--load-4bit`` / ``--load-8bit`` override ``INFER_PRECISION`` when set.
    """
    import torch

    if load_4bit:
        return False, True, torch.float16
    if load_8bit:
        return True, False, torch.float16

    prec = resolve_infer_precision()
    if prec == "4bit":
        return False, True, torch.float16
    if prec == "8bit":
        return True, False, torch.float16
    if prec in ("fp16", "float16"):
        return False, False, torch.float16
    return False, False, torch.bfloat16


def resolve_train_precision() -> str:
    """``config/run_config.TRAIN_PRECISION``: bf16 | fp16 | 8bit | 4bit."""
    try:
        from config import run_config  # type: ignore

        return str(getattr(run_config, "TRAIN_PRECISION", "8bit")).strip().lower()
    except Exception:
        return "8bit"


def train_precision_cli_args(*, precision: str | None = None) -> list[str]:
    """
    Map training precision to ``train_mem.py`` CLI flags.

    8bit/4bit use bitsandbytes on the LLM plus ``--bf16`` for compute / LoRA (see ``setup_quantization``).
    """
    prec = (precision or resolve_train_precision()).strip().lower()
    if prec == "4bit":
        return ["--bits", "4", "--bf16", "True"]
    if prec == "8bit":
        return ["--bits", "8", "--bf16", "True"]
    if prec in ("fp16", "float16"):
        return ["--fp16", "True"]
    if prec in ("bf16", "bfloat16"):
        return ["--bf16", "True"]
    raise ValueError(
        f"Unknown train precision {prec!r}; use bf16, fp16, 8bit, or 4bit "
        "(config/run_config.TRAIN_PRECISION or run.py train --train-precision)."
    )


def get_load_vision_tower_separately(backbone_id: str | None = None) -> bool:
    """InternVL: load InternViT weights from VISION_TOWER_PATH. LLaVA always uses separate CLIP tower."""
    bid = resolve_backbone_id(backbone_id)
    if bid != "internvl":
        return True
    mod = _load_paths_module(bid)
    return bool(getattr(mod, "LOAD_VISION_TOWER_SEPARATELY", True))


def get_checkpoint_suffix(backbone_id: str | None = None) -> str:
    cfg = get_backbone_config(backbone_id)
    return str(getattr(cfg, "CHECKPOINT_SUFFIX", "_llava"))


def get_train_backbone_flags(backbone_id: str | None = None) -> dict[str, str]:
    cfg = get_backbone_config(backbone_id)
    flags = getattr(cfg, "TRAIN_BACKBONE_FLAGS", None)
    return dict(flags) if isinstance(flags, dict) else {}


def get_routing_feature_dim(backbone_id: str | None = None) -> int:
    cfg = get_backbone_config(backbone_id)
    return int(getattr(cfg, "ROUTING_FEATURE_DIM", getattr(cfg, "CLIP_FEATURE_DIM", 768)))


def get_num_hidden_layers(backbone_id: str | None = None) -> int:
    cfg = get_backbone_config(backbone_id)
    return int(getattr(cfg, "NUM_HIDDEN_LAYERS", 32))


def get_last_lora_block_index(backbone_id: str | None = None) -> int:
    cfg = get_backbone_config(backbone_id)
    if hasattr(cfg, "LAST_LORA_BLOCK_INDEX"):
        return int(cfg.LAST_LORA_BLOCK_INDEX)
    return get_num_hidden_layers(backbone_id) - 1


def get_default_conv_mode(backbone_id: str | None = None) -> str:
    cfg = get_backbone_config(backbone_id)
    return str(getattr(cfg, "DEFAULT_CONV_MODE", "vicuna_v1"))


def get_backbone_id(backbone_id: str | None = None) -> str:
    cfg = get_backbone_config(backbone_id)
    return str(getattr(cfg, "BACKBONE_ID", resolve_backbone_id(backbone_id)))


def import_mllm_model_class(backbone_id: str | None = None):
    bid = resolve_backbone_id(backbone_id)
    if bid == "internvl":
        from backbone.internvl.model import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    from backbone.llava.model import LlavaLlamaForCausalLM

    return LlavaLlamaForCausalLM


def is_multimodal_model_name(model_name: str, backbone_id: str | None = None) -> bool:
    name = (model_name or "").lower()
    bid = resolve_backbone_id(backbone_id)
    if bid == "internvl":
        return "intern" in name or "llava" in name
    return "llava" in name


def apply_checkpoint_suffix_to_task(task: dict[str, Any], suffix: str) -> dict[str, Any]:
    """Rewrite benchmark task paths from legacy ``_llava`` to active suffix."""
    out = dict(task)
    legacy = "_llava"

    def _swap(path: Any) -> Any:
        if not isinstance(path, str):
            return path
        if legacy in path and suffix != legacy:
            return path.replace(legacy, suffix)
        return path

    if "output_dir" in out:
        out["output_dir"] = _swap(out["output_dir"])
    if out.get("previous_task"):
        out["previous_task"] = _swap(out["previous_task"])
    return out
