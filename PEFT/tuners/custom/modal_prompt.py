# -*- encoding: utf-8 -*-
"""
ModalPrompt PEFT tuner: per-task soft prompts with per-task prompt transforms.

Training: only the current task's prompt parameters and transform are trainable.
Inference: the integration sets ``selected_prompt_indices`` to select which task
prompts to prepend (top-K from modal guidance).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...utils import PeftType, PeftConfig


@dataclass
class ModalPromptConfig(PeftConfig):
    """PEFT config: per-task soft prompts + prompt transforms for ModalPrompt."""

    num_tasks: int = field(default=8, metadata={"help": "Number of continual tasks / prompt slots"})
    prefix_len: int = field(default=10, metadata={"help": "Soft prompt length per task (virtual tokens)"})
    hidden_size: int = field(default=4096, metadata={"help": "LLM hidden size"})
    feature_dim: int = field(default=768, metadata={"help": "CLIP feature dimension for prompt transforms"})
    cur_task: int = field(default=0, metadata={"help": "Active task id during training"})

    def __post_init__(self):
        self.peft_type = PeftType.MODAL_PROMPT
        if self.task_type is None:
            self.task_type = "CAUSAL_LM_MODAL_PROMPT"


class ModalPromptModel(nn.Module):
    """
    Wraps the HF causal LM: prepends task-specific soft prompts to ``inputs_embeds``.

    Supports multi-prompt prepending: the integration sets ``selected_prompt_indices``
    to a list of task indices whose prompts should be concatenated and prepended.
    """

    def __init__(self, model: nn.Module, config: Dict[str, ModalPromptConfig], adapter_name: str):
        super().__init__()
        self.peft_config = config
        self.active_adapter = adapter_name
        cfg = self.peft_config[adapter_name]

        inner_cfg = getattr(model, "config", None)
        if inner_cfg is None:
            hidden = int(cfg.hidden_size)
        else:
            hidden = int(
                getattr(inner_cfg, "hidden_size", None)
                or getattr(inner_cfg, "d_model", cfg.hidden_size)
            )

        self.prefix_len = int(cfg.prefix_len)
        self.num_tasks = int(cfg.num_tasks)
        self.hidden_size = hidden
        self.feature_dim = int(cfg.feature_dim)

        # Per-task learnable soft prompt parameters
        # Shape: (num_tasks, prefix_len, hidden_size)
        self.task_prompts = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.prefix_len, hidden) * 0.02)
                for _ in range(self.num_tasks)
            ]
        )

        # Per-task prompt transforms: Linear -> SiLU -> Linear (hidden -> hidden -> feature_dim)
        self.prompt_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, self.feature_dim),
                )
                for _ in range(self.num_tasks)
            ]
        )

        # Set by integration before each forward: list of task indices to prepend
        self.selected_prompt_indices: List[int] = []

        self.add_module("model", model)
        self.config = getattr(model, "config", {"model_type": "custom"})

    def _active_task_id(self, training: bool) -> int:
        cfg = self.peft_config[self.active_adapter]
        if training:
            return int(getattr(cfg, "cur_task", 0))
        return int(getattr(cfg, "cur_task", 0))

    def _prompt_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build combined prompt from selected_prompt_indices. Falls back to cur_task if none selected."""
        indices = self.selected_prompt_indices
        if not indices:
            tid = self._active_task_id(self.training)
            tid = max(0, min(tid, self.num_tasks - 1))
            indices = [tid]

        prompts = []
        for tid in indices:
            tid = max(0, min(int(tid), self.num_tasks - 1))
            p = self.task_prompts[tid].to(device=device, dtype=dtype)
            prompts.append(p.unsqueeze(0).expand(batch_size, -1, -1))
        return torch.cat(prompts, dim=1)  # (bs, total_prompt_len, hidden)

    def _total_prompt_len(self) -> int:
        indices = self.selected_prompt_indices
        if not indices:
            return self.prefix_len
        return len(indices) * self.prefix_len

    def set_trainable_prompts(self, only_task_id: Optional[int]) -> None:
        """Set which task's prompts are trainable. None = freeze all."""
        for i, p in enumerate(self.task_prompts):
            p.requires_grad = only_task_id is not None and i == int(only_task_id)

    def set_trainable_transforms(self, only_task_id: Optional[int]) -> None:
        """Set which task's transform is trainable. None = freeze all."""
        for i, t in enumerate(self.prompt_transforms):
            for p in t.parameters():
                p.requires_grad = only_task_id is not None and i == int(only_task_id)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ):
        # Incremental decode: prompts already consumed in cache
        if past_key_values is not None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if inputs_embeds is None and input_ids is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if inputs_embeds is None:
            emb = self.model.get_input_embeddings()
            inputs_embeds = emb(input_ids)
            input_ids = None

        b, _, h = inputs_embeds.shape
        prompts = self._prompt_batch(b, inputs_embeds.device, inputs_embeds.dtype)
        total_prompt_len = prompts.shape[1]
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        if attention_mask is not None:
            prefix = torch.ones(b, total_prompt_len, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix, attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full(
                (b, total_prompt_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([ignore, labels], dim=1)

        if position_ids is not None:
            warnings.warn("[ModalPrompt] position_ids are ignored when prepending soft prompts.")
            position_ids = None

        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        try:
            inner = super().__getattribute__("_modules")["model"]
        except (AttributeError, KeyError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None
        return getattr(inner, name)
