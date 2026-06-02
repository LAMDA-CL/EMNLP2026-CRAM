"""Patch ``torch.utils.checkpoint.checkpoint`` to default ``use_reentrant=False``.

Transformers LLaMA with gradient checkpointing uses HF checkpoint with ``use_reentrant=True`` by default,
which can re-enter the same params during segmented backward; DeepSpeed ZeRO-2 may then hit
``already been reduced`` on hooks.

``use_reentrant=False`` uses a different backward path and usually avoids that nesting.
Falls back automatically on older PyTorch without the keyword."""

from __future__ import annotations

_PATCHED = False


def apply_gradient_checkpoint_use_reentrant_false() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import torch.utils.checkpoint as cu

    _orig = cu.checkpoint

    def _checkpoint(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        try:
            return _orig(function, *args, **kwargs)
        except TypeError:
            kwargs.pop("use_reentrant", None)
            return _orig(function, *args, **kwargs)

    cu.checkpoint = _checkpoint  # type: ignore[assignment]
    _PATCHED = True
