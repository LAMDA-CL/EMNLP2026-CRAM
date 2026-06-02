# -*- encoding: utf-8 -*-
"""
Path-level PEFT filtering on multimodal LLaVA-style models: skip subtrees when any path segment matches.

``exclude_module_path_segments`` (see ``PeftConfig``):

- ``None``: built-in LLaVA defaults (CLIP towers + ``mm_projector`` + ``vision_resampler``).
- ``[]``: no path filtering; only ``target_modules`` rules apply (may inject into CLIP, etc.).
- Non-empty ``list``: skip when any segment name appears in the module path (per-method policy).
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

# Common non-decoder branches under ``LlavaMetaModel`` / ``LlavaLlamaModel`` that share names with LM layers
DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS: FrozenSet[str] = frozenset(
    {
        "text_tower",
        "vision_tower",
        "mm_projector",
        "vision_resampler",
    }
)


def should_skip_peft_path(module_key: str, exclude_module_path_segments: Optional[Sequence[str]]) -> bool:
    """
    Return True if PEFT must not inject under ``module_key``.

    Matches ``PeftConfig.exclude_module_path_segments`` / training ``ModelArguments``.
    """
    if not module_key:
        return False
    if exclude_module_path_segments is None:
        active: FrozenSet[str] = DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS
    elif len(exclude_module_path_segments) == 0:
        return False
    else:
        active = frozenset(exclude_module_path_segments)
    parts = module_key.split(".")
    return any(p in active for p in parts)


def should_skip_peft_on_llava_path(module_key: str) -> bool:
    """Legacy helper: same as ``exclude_module_path_segments is None`` (LLaVA default skips only)."""
    return should_skip_peft_path(module_key, None)
