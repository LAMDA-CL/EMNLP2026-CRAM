from __future__ import annotations

from typing import Any, List

import torch

from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.tuners.custom.moelora import MoELoRAConfig, MoELoRAModel

    register_peft_extension(
        peft_type="MOE_LORA_MOELORA",
        config_cls=MoELoRAConfig,
        tuner_model_cls=MoELoRAModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("moelora")
class MoeloraIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))

    def initialize_model(self, model) -> None:
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_moelora(model)

    def _setup_moelora(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.moelora import MoELoRAConfig

        target_modules = self._find_target_modules(model)

        r = int(getattr(self.config, "lora_r", 64))
        if r % self.task_num != 0:
            raise ValueError(
                f"lora_r={r} must be divisible by task_num={self.task_num} (MoELoRA rank split)."
            )

        peft_config = MoELoRAConfig(
            target_modules=target_modules,
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.task_num,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, peft_config)

    def _find_target_modules(self, model) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)
