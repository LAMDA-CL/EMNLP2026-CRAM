# method/base/hooks.py
"""Forward hook registry; passes optional ``CLContext`` without cluttering backbone code."""
import logging
import torch.nn as nn
from typing import Callable, Dict, List, Optional
from .context import CLContext


class HookManager:
    """Register and remove forward hooks; optional ``CLContext`` forwarded into hook callbacks."""

    def __init__(self):
        self.handles: List[nn.modules.hooks.RemovableHandle] = []
        self.hook_registry: Dict[str, Callable] = {}

    def register_layer_hook(
        self,
        model: nn.Module,
        layer_path: str,
        hook_fn: Callable,
        context: Optional[CLContext] = None,
    ) -> None:
        """Register a hook on ``layer_path`` (e.g. ``model.layers.10``). Callback signature: ``(module, input, output[, context])``."""
        try:
            layer = dict(model.named_modules())[layer_path]

            def wrapped_hook(module, input, output):
                if context is not None:
                    return hook_fn(module, input, output, context)
                else:
                    return hook_fn(module, input, output)

            handle = layer.register_forward_hook(wrapped_hook)
            self.handles.append(handle)
            self.hook_registry[layer_path] = hook_fn

        except KeyError:
            logging.warning("HookManager: layer %s not found in model.", layer_path)

    def register_module_hook(
        self,
        module: nn.Module,
        hook_fn: Callable,
        context: Optional[CLContext] = None,
    ) -> None:
        """Register a forward hook on ``module``."""

        def wrapped_hook(module, input, output):
            if context is not None:
                return hook_fn(module, input, output, context)
            else:
                return hook_fn(module, input, output)

        handle = module.register_forward_hook(wrapped_hook)
        self.handles.append(handle)

    def remove_all(self) -> None:
        """Remove every registered hook."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hook_registry.clear()

    def remove_hook(self, layer_path: str) -> None:
        """Remove hook for ``layer_path`` (implemented as remove-all for simplicity)."""
        if layer_path in self.hook_registry:
            self.remove_all()
            del self.hook_registry[layer_path]

    def get_registered_hooks(self) -> Dict[str, Callable]:
        """Copy of ``layer_path -> hook_fn`` registry."""
        return self.hook_registry.copy()
