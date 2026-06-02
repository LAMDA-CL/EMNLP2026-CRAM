# method/base/context.py
"""Per-forward-step CL context (aux losses, task id) without mutating the backbone module."""
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class CLContext:
    is_training: bool = True
    task_id: Optional[int] = None
    step_count: int = 0

    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        """Clear step-scoped caches (e.g. auxiliary losses)."""
        self.auxiliary_losses.clear()

    def add_auxiliary_loss(self, name: str, loss: torch.Tensor):
        self.auxiliary_losses[name] = loss

    def get_total_auxiliary_loss(self) -> torch.Tensor:
        if not self.auxiliary_losses:
            return torch.tensor(0.0)
        return sum(self.auxiliary_losses.values())
