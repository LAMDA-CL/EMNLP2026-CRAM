# method/base/integration.py
"""
Abstract integration layer between CL methods and the training/inference stack.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch.nn as nn
import torch

from .context import CLContext


class CLIntegration(ABC):
    """Lifecycle hooks for a continual-learning method (hooks, losses, save/load)."""

    def __init__(self, config: Any):
        self.config = config

    @property
    def peft_exclude_module_path_segments(self) -> Optional[List[str]]:
        """
        Same semantics as ``PeftConfig.exclude_module_path_segments``:
        ``None`` = LLaVA default skip list; ``[]`` = no filtering; non-empty = custom path segments.
        Set via ``exclude_module_path_segments`` in ``config/methods/<method>.py`` ``METHOD_CONFIG``.
        """
        return getattr(self.config, "exclude_module_path_segments", None)

    @abstractmethod
    def initialize_model(self, model: nn.Module) -> None:
        """After model build: register hooks, buffers, PEFT, etc."""
        pass

    @abstractmethod
    def on_input_prep(
        self,
        model: nn.Module,
        args: tuple,
        kwargs: dict,
        context: CLContext,
    ) -> None:
        """Before multimodal input prep (e.g. routing, ``task_id`` on adapters)."""
        pass

    @abstractmethod
    def on_forward_start(
        self,
        model: nn.Module,
        context: CLContext,
    ) -> None:
        """Start of forward: reset step context."""
        pass

    @abstractmethod
    def on_forward_end(
        self,
        model: nn.Module,
        outputs: Any,
        context: CLContext,
    ) -> Any:
        """End of forward: attach aux losses or adjust ``outputs``."""
        pass

    def on_step_end(
        self,
        model: nn.Module,
        context: CLContext,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        """After optimizer step (via ``CLTrainerCallback``); default no-op."""
        return

    @abstractmethod
    def on_task_end(
        self,
        model: nn.Module,
        context: CLContext,
        task_id: int,
    ) -> None:
        """When a task segment finishes (freeze buffers, save stats, etc.)."""
        pass

    @abstractmethod
    def get_inference_config(self) -> Dict:
        """Extra kwargs or flags for inference."""
        pass

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        """
        Persist non-PEFT method state (prototypes, anchors, Gram, ...). Default: skip.

        Returns:
            Whether save ran successfully (base returns ``False``).
        """
        return False

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        """
        Load method-specific state. Default: skip. ``model`` matches ``common/load_model.py`` usage.

        Returns:
            Whether load ran successfully.
        """
        return False

    def compute_total_loss(
        self,
        base_loss: torch.Tensor,
        context: CLContext,
    ) -> torch.Tensor:
        """``base_loss`` plus auxiliary terms from ``context``."""
        return base_loss + context.get_total_auxiliary_loss()

    def prepare_training_data(self, data_args: Any, model_args: Any, training_args: Any = None) -> None:
        """
        Called before ``make_supervised_data_module`` (e.g. replay writes sidecar JSON / ``memory_data_path``).
        """
        return

    def on_training_batch_end(
        self,
        model: nn.Module,
        context: CLContext,
        batch: Dict[str, Any],
        *,
        loss: Optional[torch.Tensor] = None,
        trainer: Any = None,
    ) -> None:
        """
        After backward per micro-batch (gradient accumulation included).
        ``batch`` matches ``compute_loss``; ``cl_raw_example`` may hold raw JSON-aligned rows for replay.
        """
        return

    def should_store_training_example(
        self,
        model: nn.Module,
        context: CLContext,
        raw_example: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        example_index: int,
        loss: Optional[torch.Tensor] = None,
    ) -> bool:
        """Whether to push ``raw_example`` into an in-training buffer (e.g. replay). Default ``False``."""
        return False
