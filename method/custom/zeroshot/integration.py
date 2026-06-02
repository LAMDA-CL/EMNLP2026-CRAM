from __future__ import annotations

from typing import Any, Dict

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


@CLMethodFactory.register("zeroshot")
class ZeroshotIntegration(CLIntegration):
    def initialize_model(self, model) -> None:
        return

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict:
        return {}
