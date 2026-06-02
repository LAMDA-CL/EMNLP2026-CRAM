"""Shared inference tower paths for continual-learning methods (not backbone paths)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

from config.paths.llava_paths import CLIP_PATH

# Default CLIP snapshot for routing / text towers when a method enables them in INFER_DEFAULTS.
DEFAULT_ROUTING_VISION_TOWER_PATH = CLIP_PATH
DEFAULT_TEXT_TOWER_PATH = CLIP_PATH

_METHODS_DIR = Path(__file__).resolve().parent.parent / "config" / "methods"


def _normalize_tower_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    s = str(path).strip()
    if not s or s.lower() in ("none", "false", "0"):
        return None
    import os

    return os.path.normpath(os.path.realpath(os.path.expanduser(s)))


def tower_paths_equal(a: Optional[str], b: Optional[str]) -> bool:
    """True if both paths resolve to the same on-disk directory."""
    pa = _normalize_tower_path(a)
    pb = _normalize_tower_path(b)
    if pa is None or pb is None:
        return False
    return pa == pb


def should_load_routing_vision_tower(
    routing_vision_tower: Optional[str],
    mllm_vision_tower: Optional[str],
) -> bool:
    """Skip a second tower when routing uses the same weights/path as the MLLM vision encoder."""
    if not _normalize_tower_path(routing_vision_tower):
        return False
    return not tower_paths_equal(routing_vision_tower, mllm_vision_tower)


_NAME_TO_TOWER_PATH = {
    "DEFAULT_ROUTING_VISION_TOWER_PATH": DEFAULT_ROUTING_VISION_TOWER_PATH,
    "DEFAULT_TEXT_TOWER_PATH": DEFAULT_TEXT_TOWER_PATH,
    "CLIP_PATH": CLIP_PATH,
}

_AST_TOWER_ATTRS = {
    "infer_paths.DEFAULT_ROUTING_VISION_TOWER_PATH",
    "config.paths.infer_paths.DEFAULT_ROUTING_VISION_TOWER_PATH",
    "utils.infer_paths.DEFAULT_ROUTING_VISION_TOWER_PATH",
    "infer_paths.DEFAULT_TEXT_TOWER_PATH",
    "config.paths.infer_paths.DEFAULT_TEXT_TOWER_PATH",
    "utils.infer_paths.DEFAULT_TEXT_TOWER_PATH",
}


def _ast_infer_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _NAME_TO_TOWER_PATH:
            return _NAME_TO_TOWER_PATH[node.id]
        if node.id in ("True", "False"):
            return node.id == "True"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        qual = f"{node.value.id}.{node.attr}"
        if qual in _AST_TOWER_ATTRS and qual.endswith("DEFAULT_ROUTING_VISION_TOWER_PATH"):
            return DEFAULT_ROUTING_VISION_TOWER_PATH
        if qual in _AST_TOWER_ATTRS and qual.endswith("DEFAULT_TEXT_TOWER_PATH"):
            return DEFAULT_TEXT_TOWER_PATH
    return None


def _read_infer_defaults_from_method_py(method: str) -> dict[str, Any]:
    """Parse INFER_DEFAULTS without importing the method module (avoids PEFT/transformers deps)."""
    path = _METHODS_DIR / f"{method}.py"
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "INFER_DEFAULTS":
                if not isinstance(node.value, ast.Dict):
                    return {}
                out: dict[str, Any] = {}
                for key_node, val_node in zip(node.value.keys, node.value.values):
                    if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                        continue
                    out[key_node.value] = _ast_infer_value(val_node)
                return out
    return {}


def resolve_method_infer_tower_paths(method: Optional[str]) -> dict[str, Optional[str]]:
    """
    Read ``routing_vision_tower`` / ``text_tower`` from ``config/methods/<method>.py`` INFER_DEFAULTS.

    Missing keys mean do not load that tower at inference.
    """
    m = (method or "").strip().lower()
    if not m or m in ("base",):
        return {"routing_vision_tower": None, "text_tower": None}

    routing: Optional[str] = None
    text: Optional[str] = None
    infer = _read_infer_defaults_from_method_py(m)
    if not infer:
        try:
            mod = __import__(f"config.methods.{m}", fromlist=["INFER_DEFAULTS"])
            infer = getattr(mod, "INFER_DEFAULTS", None) or {}
        except Exception:
            infer = {}

    if isinstance(infer, dict):
        if "routing_vision_tower" in infer:
            routing = infer.get("routing_vision_tower")
            if routing is True or routing == "__default__":
                routing = DEFAULT_ROUTING_VISION_TOWER_PATH
        if "text_tower" in infer:
            text = infer.get("text_tower")
            if text is True or text == "__default__":
                text = DEFAULT_TEXT_TOWER_PATH

    return {
        "routing_vision_tower": _normalize_tower_path(routing) if routing is not None else None,
        "text_tower": _normalize_tower_path(text) if text is not None else None,
    }
