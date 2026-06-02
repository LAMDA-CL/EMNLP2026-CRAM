"""
Aligns with ``PEFT.utils.llava_peft_scope`` so method code scanning ``named_modules`` uses the same
``exclude_module_path_segments`` rules as PEFT injection.
"""

from __future__ import annotations

import logging
from typing import Any, List

import torch.nn as nn

from PEFT.utils.llava_peft_scope import DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS, should_skip_peft_path
from PEFT.utils.peft_target_modules import resolve_peft_target_spec_to_allowed_suffixes
from utils.rank import is_local_main_process


def should_skip_module_for_peft_scan(full_module_name: str, method_config: Any) -> bool:
    segs = getattr(method_config, "exclude_module_path_segments", None)
    return should_skip_peft_path(full_module_name, segs)


def module_path_outside_llm_for_peft(full_module_name: str) -> bool:
    return should_skip_peft_path(full_module_name, None)


module_path_under_clip_tower = module_path_outside_llm_for_peft


def collect_peft_target_linear_suffixes(model: nn.Module, method_config: Any) -> List[str]:
    _root = getattr(model, "_base_model", None) or model
    found: set = set()
    for name, module in _root.named_modules():
        if should_skip_module_for_peft_scan(name, method_config):
            continue
        if not isinstance(module, nn.Linear):
            continue
        found.add(name.split(".")[-1])
    spec = getattr(method_config, "peft_target_modules", None)
    if spec is None:
        spec = getattr(method_config, "lora_target_modules", None)
    if spec is None:
        spec = "attention"
    allow = resolve_peft_target_spec_to_allowed_suffixes(spec)
    if allow is None:
        out = sorted(found)
        if is_local_main_process():
            if not out:
                logging.warning("No in-scope nn.Linear for PEFT; check exclude_module_path_segments / model")
            else:
                logging.info("PEFT target_modules (all in-scope Linears, spec=%r): %s", spec, out)
        return out
    out = sorted(found & allow)
    if not out and found:
        raise ValueError(
            f"peft_target_modules={spec!r} has no overlap with in-scope Linear suffixes in this model."
            f" Found suffix sample={sorted(found)[:32]}, allowed={allow}"
        )
    if not out and not found:
        raise ValueError("No in-scope nn.Linear found; check exclude_module_path_segments / model.")
    if is_local_main_process():
        logging.info("PEFT target_modules (spec=%r): %s", spec, out)
    return out


collect_lora_target_linear_suffixes = collect_peft_target_linear_suffixes

__all__ = [
    "DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS",
    "should_skip_module_for_peft_scan",
    "module_path_outside_llm_for_peft",
    "module_path_under_clip_tower",
    "collect_peft_target_linear_suffixes",
    "collect_lora_target_linear_suffixes",
]
