"""Resolve CLIP routing vision tower and extract 768-d image features for CL methods."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)) and obj:
        return obj[0]
    return obj


def _model_chain(model: Any):
    seen = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = getattr(cur, "_base_model", None)
        if nxt is cur or nxt is None:
            if hasattr(cur, "base_model") and getattr(cur, "base_model", None) is not cur:
                nxt = getattr(cur, "base_model", None)
            elif hasattr(cur, "model") and getattr(cur, "model", None) is not cur:
                nxt = getattr(cur, "model", None)
            else:
                break
        cur = nxt


def resolve_routing_vision_tower(model: Any) -> Optional[Any]:
    for m in _model_chain(model):
        rvt = _unwrap(getattr(m, "routing_vision_tower", None))
        if rvt is not None and getattr(rvt, "is_loaded", True):
            return rvt
        getter = getattr(m, "get_routing_vision_tower", None)
        if callable(getter):
            rvt = _unwrap(getter())
            if rvt is not None and getattr(rvt, "is_loaded", True):
                return rvt
    return None


def resolve_mllm_vision_tower(model: Any) -> Optional[Any]:
    for m in _model_chain(model):
        vt = _unwrap(getattr(m, "vision_tower", None))
        if vt is not None and getattr(vt, "is_loaded", True):
            return vt
        getter = getattr(m, "get_vision_tower", None)
        if callable(getter):
            vt = _unwrap(getter())
            if vt is not None and getattr(vt, "is_loaded", True):
                return vt
    return None


def extract_routing_image_features(model: Any, images: torch.Tensor) -> torch.Tensor:
    """
    Return batch of CLIP routing image features (typically [B, 768]).

    Uses ``routing_vision_tower`` when present; otherwise legacy LLaVA dual-output
    ``vision_tower`` (first tuple element = CLIP image_embeds).
    """
    rvt = resolve_routing_vision_tower(model)
    if rvt is not None:
        with torch.no_grad():
            out = rvt(images)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 3:
                return out.mean(dim=1)
            return out

    vt = resolve_mllm_vision_tower(model)
    if vt is None or not getattr(vt, "is_loaded", False):
        raise RuntimeError("routing_vision_tower not loaded and no legacy vision_tower fallback")

    with torch.no_grad():
        raw = vt(images)
        if isinstance(raw, tuple):
            raw = raw[0]
        if raw.dim() == 3:
            return raw.mean(dim=1)
        return raw


def resolve_text_tower(model: Any) -> Optional[Any]:
    for m in _model_chain(model):
        tt = _unwrap(getattr(m, "text_tower", None))
        if tt is not None:
            return tt
        getter = getattr(m, "get_text_tower", None)
        if callable(getter):
            tt = _unwrap(getter())
            if tt is not None:
                return tt
    return None


def resolve_clip_tokenizer(model: Any) -> Optional[Any]:
    for m in _model_chain(model):
        tok = getattr(m, "clip_tokenizer", None)
        if tok is not None:
            return tok
    return None
