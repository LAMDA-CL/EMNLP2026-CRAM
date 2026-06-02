from __future__ import annotations

from typing import Any, Dict
import torch
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return
    from PEFT.tuners.custom.olora import OLoRAConfig, OLoRAModel

    register_peft_extension(
        peft_type="MOE_LORA_OLORA",
        config_cls=OLoRAConfig,
        tuner_model_cls=OLoRAModel,
    )
    _PEFT_EXT_REGISTERED = True


def _unwrap_peft_root(model: Any) -> Any:
    return getattr(model, "_base_model", model)


def _get_olora_tuner_module(root: Any) -> Any:
    inner = getattr(root, "base_model", None)
    if inner is not None and inner.__class__.__name__ == "OLoRAModel":
        return inner
    return None


@CLMethodFactory.register("olora")
class OloraIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))
        self.olora_lambda: float = float(getattr(config, "olora_lambda", 0.5))
        self._olora_warned_zero_orth: bool = False
        self._model_ref = None

    def initialize_model(self, model) -> None:
        self._model_ref = model
        for _, p in model.named_parameters():
            p.requires_grad = False
        self._setup_olora_peft(model)
        root = _unwrap_peft_root(model)
        from PEFT.tuners.custom.olora import apply_olora_expert_trainable_mask, sync_olora_cur_task

        apply_olora_expert_trainable_mask(root, self.cur_task, adapter_name="default")
        sync_olora_cur_task(root, self.cur_task)

    def _setup_olora_peft(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.olora import OLoRAConfig

        target_modules = collect_peft_target_linear_suffixes(model, self.config)
        r = int(getattr(self.config, "lora_r", 64))
        if r % self.task_num != 0:
            raise ValueError(
                f"lora_r={r} must be divisible by task_num={self.task_num} for O-LoRA rank splitting."
            )

        peft_config = OLoRAConfig(
            target_modules=target_modules,
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", r * 2)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.task_num,
            cur_task=self.cur_task,
            task_type="CAUSAL_LM",
            bias="none",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base = getattr(model, "_base_model", None)
        if _base is not None:
            peft_model = get_peft_model(_base, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, peft_config)

        wrapped = getattr(model, "_base_model", None)
        if wrapped is not None and hasattr(wrapped, "print_trainable_parameters"):
            wrapped.print_trainable_parameters()

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model, context: CLContext) -> None:
        from PEFT.tuners.custom.olora import sync_olora_cur_task

        ct = int(getattr(context, "task_id", self.cur_task))
        sync_olora_cur_task(_unwrap_peft_root(model), ct)

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        if not model.training:
            return outputs
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return outputs

        ct = int(getattr(context, "task_id", self.cur_task))
        root = _unwrap_peft_root(model)
        tuner = _get_olora_tuner_module(root)
        if ct < 1:
            return outputs
        orth = getattr(tuner, "_olora_orth_sum", None) if tuner is not None else None
        if orth is None:
            orth = torch.zeros((), device=loss.device, dtype=loss.dtype)
            if ct >= 1 and not self._olora_warned_zero_orth:
                self._olora_warned_zero_orth = True
        orth_penalty = self.olora_lambda * orth
        outputs.loss = loss + orth_penalty

        return outputs

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict:
        return {
            "task_num": self.task_num,
            "cur_task": int(getattr(self.config, "cur_task", self.cur_task)),
            "olora_lambda": self.olora_lambda,
            "olora_orthogonal_log_interval": int(
                getattr(self.config, "olora_orthogonal_log_interval", 0)
            ),
        }

    def compute_total_loss(self, base_loss: torch.Tensor, context: CLContext) -> torch.Tensor:
        return base_loss + context.get_total_auxiliary_loss()
