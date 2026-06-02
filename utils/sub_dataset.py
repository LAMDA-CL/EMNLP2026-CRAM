"""UCIT: add/remove ``_sub`` on JSON instruction paths (see ``run_config.use_sub_dataset``).

Used by ``run.py`` and eval scripts; not model hyperparameters."""
from __future__ import annotations

import copy
from typing import Any, Dict

# Task dict keys holding JSON file paths (only paths ending in .json are rewritten).
_TASK_JSON_PATH_KEYS = ("train_data_path", "test_data_path", "eval_annotation_path")


def json_path_with_sub_suffix(path: str) -> str:
    """``.../foo.json`` → ``.../foo_sub.json``; unchanged if already ``*_sub.json``."""
    if not path or not isinstance(path, str):
        return path
    if not path.endswith(".json"):
        return path
    if path.endswith("_sub.json"):
        return path
    return f"{path[:-5]}_sub.json"


def json_path_without_sub_suffix(path: str) -> str:
    """``.../foo_sub.json`` → ``.../foo.json``; otherwise unchanged."""
    if not path or not isinstance(path, str):
        return path
    if path.endswith("_sub.json"):
        return f"{path[:-9]}.json"
    return path


def apply_use_sub_dataset_to_task(
    task: Dict[str, Any],
    *,
    use_sub_dataset: bool,
    benchmark: str,
) -> Dict[str, Any]:
    """
    Only when ``benchmark == ucit`` rewrite paths (CoIN etc. keep canonical json names).
    Config files should use names **without** ``_sub``; with ``use_sub_dataset=True`` append ``_sub``.
    """
    t = copy.deepcopy(task)
    if str(benchmark).strip().lower() != "ucit":
        return t
    for key in _TASK_JSON_PATH_KEYS:
        if key not in t or t[key] is None:
            continue
        p = t[key]
        if not isinstance(p, str):
            continue
        t[key] = json_path_with_sub_suffix(p) if use_sub_dataset else json_path_without_sub_suffix(p)
    return t
