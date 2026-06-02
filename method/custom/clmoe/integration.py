"""
CL-MoE (Continual Learning Mixture of Experts LoRA): injects MoE-LoRA layers.
The router is input-dependent (per-layer, per-token), eliminating the need for
explicit task-ID gating.  Combined with memory replay, this provides a strong
continual learning baseline.

Reference: CL-MoE (ICLR 2025) — Mixture of Experts for Continual Instruction Tuning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import should_skip_module_for_peft_scan
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.tuners.custom.clmoe import CLMoEConfig, CLMoEModel

    register_peft_extension(
        peft_type="CLMOE",
        config_cls=CLMoEConfig,
        tuner_model_cls=CLMoEModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("clmoe")
class ClmoeIntegration(CLIntegration):
    """
    CL-MoE integration for PRISM.

    Relies entirely on input-dependent routing via the per-layer routers inside
    ``CLMoELinear``.  No task-ID gating or external prototype matching is used.

    Supports memory replay via ``--memory_data_path`` (handled at the data-loader
    level by ``LazySupervisedDataset``, no extra logic needed here).
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 4)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))

    # ------------------------------------------------------------------
    #  Required CLIntegration interface
    # ------------------------------------------------------------------

    def initialize_model(self, model) -> None:
        """Freeze backbone, wrap in CL-MoE PEFT."""
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_clmoe(model)

    def on_input_prep(self, model, args, kwargs, context) -> None:
        """No-op: CL-MoE routing is input-dependent inside forward, not task-ID gated."""
        pass

    def on_forward_start(self, model, context) -> None:
        """No-op: CL-MoE has no per-step state to reset."""
        pass

    def on_forward_end(self, model, outputs, context):
        """No-op: CL-MoE does not add auxiliary losses."""
        return outputs

    def on_task_end(self, model, context, task_id) -> None:
        """Called at task boundary.  No state consolidation needed."""
        pass

    def get_inference_config(self) -> Dict:
        """Return inference-time configuration."""
        return {
            "clmethod": "clmoe",
        }

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _setup_clmoe(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.clmoe import CLMoEConfig

        target_modules = self._find_target_modules(model)

        r = int(getattr(self.config, "lora_r", 64))
        if r % self.task_num != 0:
            r = ((r // self.task_num) + 1) * self.task_num

        peft_config = CLMoEConfig(
            target_modules=target_modules,
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.task_num,
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
        """Scan LLM backbone for all Linear layers (attention + FFN), matching original CL-MoE."""
        target_modules = set()
        _base_model = getattr(model, "_base_model", None) or model
        for name, module in _base_model.named_modules():
            if should_skip_module_for_peft_scan(name, self.config):
                continue
            if isinstance(module, torch.nn.Linear) and any(
                x in name for x in (
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                )
            ):
                target_modules.add(name.split(".")[-1])
        return list(target_modules)
