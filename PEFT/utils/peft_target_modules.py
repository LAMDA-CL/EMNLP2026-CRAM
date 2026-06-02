"""
Presets and parsing for PEFT ``target_modules`` (suffix names like ``q_proj``, ``gate_proj``).

Primary key in ``config/methods/<method>.py`` ``METHOD_CONFIG``: ``peft_target_modules``.
Legacy name ``lora_target_modules`` is still accepted with the same meaning.

**Preset strings** (case-insensitive):
  - ``attention`` / ``attn``: attention Linears (typically ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``).
  - ``attn_qv`` (aliases ``attn_q_v``, ``q_v``, ``qv``, ``wq_wv``): **only** ``q_proj`` and ``v_proj``
    (O-LoRA / Hu et al.: adapt ``W_q``, ``W_v`` only).
  - ``ffn`` / ``mlp``: ``gate_proj``, ``up_proj``, ``down_proj``.
  - ``attn_and_ffn`` (aliases ``attn_ffn``, ``attn+ffn``, ``transformer``): attention ∪ FFN (**excludes**
    ``lm_head``; HiDe-style setups often use this).
  - ``linear`` / ``all`` / ``full``: after ``exclude_module_path_segments``, collect every in-scope ``nn.Linear``
    suffix (SAME-style).

**Or** a list of submodule suffix strings intersected with names present in the model.

Default when unset matches historical repo behavior: ``attention``.

CLI: ``--peft_target_modules ffn`` or comma-separated / JSON list ``["q_proj","v_proj"]``.
``lora_target_modules`` in ``METHOD_CONFIG`` still works.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Set

# Vicuna-7B / LLaMA-style naming
PEFT_TARGET_MODULE_PRESETS: dict[str, List[str]] = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ffn": ["gate_proj", "up_proj", "down_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}

# Legacy alias map (some docs say LoRA-specific names)
PEFT_LORA_TARGET_PRESETS = PEFT_TARGET_MODULE_PRESETS


def resolve_peft_target_spec_to_allowed_suffixes(raw: Any) -> Optional[Set[str]]:
    """
    Resolve ``peft_target_modules`` into allowed submodule **suffix** names for LoRA-like adapters.

    ``None`` means no filtering beyond path excludes: allow every scanned Linear suffix.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set, frozenset)):
        s = {str(x).strip() for x in raw if str(x).strip()}
        if not s:
            raise ValueError("peft_target_modules empty list is invalid; use preset 'all' or explicit names")
        return s
    s0 = str(raw).strip()
    if not s0:
        return None
    low = s0.lower()
    if low in ("all", "full", "linear", "*", "any"):
        return None
    if low == "attn":
        return set(PEFT_TARGET_MODULE_PRESETS["attention"])
    if low in ("attn_qv", "attn_q_v", "q_v", "qv", "wq_wv"):
        return {"q_proj", "v_proj"}
    if low in ("attn_and_ffn", "attn_ffn", "attn+ffn", "transformer", "transformer_blocks"):
        return set(PEFT_TARGET_MODULE_PRESETS["attention"]) | set(PEFT_TARGET_MODULE_PRESETS["ffn"])
    if low in PEFT_TARGET_MODULE_PRESETS:
        return set(PEFT_TARGET_MODULE_PRESETS[low])
    if s0.lstrip().startswith("["):
        try:
            arr = json.loads(s0)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot parse peft_target_modules as JSON list: {e}") from e
        if not isinstance(arr, (list, tuple)):
            raise ValueError("peft_target_modules JSON must be a list")
        s = {str(x).strip() for x in arr if str(x).strip()}
        if not s:
            raise ValueError("peft_target_modules JSON list is empty; use preset 'all' explicitly")
        return s
    if "," in s0 or re.search(r"\s{2,}", s0):
        return {p.strip() for p in re.split(r"[\s,]+", s0) if p.strip()}
    return {s0}


# Legacy entry point name
resolve_lora_target_spec_to_allowed_suffixes = resolve_peft_target_spec_to_allowed_suffixes


__all__ = [
    "PEFT_TARGET_MODULE_PRESETS",
    "PEFT_LORA_TARGET_PRESETS",
    "resolve_peft_target_spec_to_allowed_suffixes",
    "resolve_lora_target_spec_to_allowed_suffixes",
]
