# -*- encoding: utf-8 -*-
"""
Orthogonal Low-Rank Adaptation (O-LoRA) — multi-task LoRA.

- Each layer keeps ``expert_num`` low-rank adapter groups (one slot per sequential task, up to ``expert_num``).
- **Train**: only A/B for ``cur_task`` participate in the forward; optional orthogonal regularizer on past A
  (see ``compute_olora_orthogonal_loss``).
- **Eval**: concatenate A/B for tasks ``0..cur_task`` along rank — cumulative LoRA effect (CoIN / original OLoRA style).

No token-level router (orthogonal to ``MoELoRA`` soft routing).

Orthogonal terms are computed inside ``OLoRALinear.forward``, **fused** into ``result`` with a tiny scale, then summed in
``OLoRAModel._olora_orth_sum`` and attached to ``loss`` — same backward path as CE / checkpoint / ZeRO-2.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ...import_utils import is_bnb_4bit_available, is_bnb_available
from ...utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from ..standard.lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

if is_bnb_available():
    import bitsandbytes as bnb

# Fuse orthogonal scalar into linear output so gradients to A flow through ``result`` (scale negligible for logits)
OLORA_ORTH_FUSE_SCALE: float = 1e-8


def _olora_linear_orthogonal_fragment(linear: "OLoRALinear", adapter_name: str) -> Optional[torch.Tensor]:
    """Per layer: L1 ``sum_{i<t} |A_t A_i^T|``; historical ``A_i`` detached. None if no history."""
    if linear.cur_task < 1:
        return None
    if adapter_name not in linear.lora_A:
        return None
    la = linear.lora_A[adapter_name]
    ct = int(linear.cur_task)
    a_cur = la.loraA[ct].mlp.weight
    total: Optional[torch.Tensor] = None
    for i in range(ct):
        a_prev = la.loraA[i].mlp.weight
        term = torch.abs(torch.mm(a_cur, a_prev.detach().T)).sum()
        total = term if total is None else total + term
    return total


@dataclass
class OLoRAConfig(LoraConfig):
    """O-LoRA: standard LoRA fields plus task slots and current task index."""

    expert_num: int = field(default=8, metadata={"help": "Task slots / experts (max sequential tasks)."})
    cur_task: int = field(default=0, metadata={"help": "Current training task id (0-based)."})
    task_embedding_dim: int = field(default=64, metadata={"help": "Legacy field for old CoIN configs."})

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_OLORA


class OLoRAModel(LoraModel):
    """Replace target Linears with ``OLoRALinear`` (multi-slot + single active path in train)."""

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _prepare_olora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_olora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("OLoRAModel supports only 1 adapter with bias.")

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        self._olora_orth_sum: Optional[torch.Tensor] = None
        self._olora_orth_hook_handle = None
        self._register_olora_orth_reset_hook()

    def _register_olora_orth_reset_hook(self) -> None:
        """Reset orthogonal accumulator at each LM forward start (align with decoder forward)."""
        if getattr(self, "_olora_orth_hook_handle", None) is not None:
            return
        inner = self.model
        decoder = getattr(inner, "model", None)
        if decoder is None:
            return

        def _hook(_mod, _args):
            self._olora_orth_sum = None

        self._olora_orth_hook_handle = decoder.register_forward_pre_hook(_hook)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, layer = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, self.model.training, layer)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {lora_config.target_modules} not found in base model.")

    def _create_new_module(self, lora_config, adapter_name, target, training, layer):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": getattr(lora_config, "task_embedding_dim", 64),
            "expert_num": lora_config.expert_num,
            "cur_task": lora_config.cur_task,
            "layer_id": layer,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to True for Linear; setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} not supported.")
            kwargs["olora_tuner_parent"] = self
            new_module = OLoRALinear(
                adapter_name,
                in_features,
                out_features,
                bias=bias,
                train_signal=training,
                **kwargs,
            )
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name, _ = _get_submodules(self.model, key)
            except IndexError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model


class OLoRALayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, expert_num: int, cur_task: int, training: bool, layer_id: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.training = training
        self.layer_id = layer_id

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {
                        adapter_name: OLoRALinearA(
                            self.in_features,
                            r,
                            self.expert_num,
                            self.cur_task,
                            self.layer_id,
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: OLoRALinearB(
                            r,
                            self.out_features,
                            self.expert_num,
                            self.cur_task,
                            self.layer_id,
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class OLoRALinear(nn.Linear, OLoRALayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        train_signal: bool = False,
        layer_id: int = 0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", 8)
        kwargs.pop("task_embedding_dim", None)
        olora_tuner_parent = kwargs.pop("olora_tuner_parent", None)
        self.cur_task = kwargs.pop("cur_task", 0)
        self.layer_id = layer_id
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        OLoRALayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
            cur_task=self.cur_task,
            training=train_signal,
            layer_id=layer_id,
        )

        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        # OLoRAModel is nn.Module; do not assign to self before Module.__init__ completes (would register as child)
        object.__setattr__(self, "_olora_tuner_parent", olora_tuner_parent)

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x_d = self.lora_dropout[self.active_adapter](x)
            if self.training:
                lora_a_out = self.lora_A[self.active_adapter](x_d, None)
                lora_b_out = self.lora_B[self.active_adapter](lora_a_out, None)
            else:
                lora_a_out = self.lora_A[self.active_adapter](x_d, self.cur_task)
                lora_b_out = self.lora_B[self.active_adapter](lora_a_out, self.cur_task)
            result = result + lora_b_out * self.scaling[self.active_adapter]
            if self._olora_tuner_parent is not None and self.training and self.cur_task >= 1:
                frag = _olora_linear_orthogonal_fragment(self, self.active_adapter)
                if frag is not None:
                    tp = self._olora_tuner_parent
                    tp._olora_orth_sum = frag if tp._olora_orth_sum is None else tp._olora_orth_sum + frag
                    fuse = result.new_tensor(OLORA_ORTH_FUSE_SCALE)
                    result = result + fuse * frag * torch.ones_like(result)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result


class OLoRALinearA(nn.Module):
    """LoRA A bank per task slot; training uses only ``cur_task``."""

    def __init__(self, in_features, r_total, expert_num, cur_task, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.layer_id = layer_id
        r_per = r_total // expert_num
        if r_per * expert_num != r_total:
            raise ValueError(f"LoRA rank r={r_total} must be divisible by expert_num={expert_num}.")
        self.r_per = r_per
        self.in_features = in_features
        self.loraA = nn.ModuleList([OLoRAExpert(in_features, r_per) for _ in range(expert_num)])
        # Inference: cache concatenated weights per (eval_max_task, device, dtype) to avoid per-step nn.Linear in AR decode
        self._eval_weight_cache: Dict[tuple, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, eval_max_task: Optional[int]) -> torch.Tensor:
        if eval_max_task is None:
            assert 0 <= self.cur_task < self.expert_num
            return self.loraA[self.cur_task](x)
        cache_key = (int(eval_max_task), x.device, x.dtype)
        w = self._eval_weight_cache.get(cache_key)
        if w is None:
            parts = [self.loraA[i].mlp.weight for i in range(eval_max_task + 1)]
            w = torch.cat(parts, dim=0).to(device=x.device, dtype=x.dtype)
            self._eval_weight_cache[cache_key] = w
        return F.linear(x, w)


class OLoRALinearB(nn.Module):
    """LoRA B bank per task slot."""

    def __init__(self, r_total, out_features, expert_num, cur_task, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.layer_id = layer_id
        r_per = r_total // expert_num
        if r_per * expert_num != r_total:
            raise ValueError(f"LoRA rank r={r_total} must be divisible by expert_num={expert_num}.")
        self.r_per = r_per
        self.out_features = out_features
        self.loraB = nn.ModuleList([OLoRAExpert(r_per, out_features) for _ in range(expert_num)])
        self._eval_weight_cache: Dict[tuple, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, eval_max_task: Optional[int]) -> torch.Tensor:
        if eval_max_task is None:
            assert 0 <= self.cur_task < self.expert_num
            return self.loraB[self.cur_task](x)
        cache_key = (int(eval_max_task), x.device, x.dtype)
        w = self._eval_weight_cache.get(cache_key)
        if w is None:
            parts = [self.loraB[i].mlp.weight for i in range(eval_max_task + 1)]
            w = torch.cat(parts, dim=1).to(device=x.device, dtype=x.dtype)
            self._eval_weight_cache[cache_key] = w
        return F.linear(x, w)


class OLoRAExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.mlp(x)


def sync_olora_cur_task(model: nn.Module, cur_task: int) -> None:
    """Propagate ``cur_task`` to every ``OLoRALinear`` and A/B children (train vs eval merge range)."""
    for m in model.modules():
        if isinstance(m, OLoRALinear):
            m.cur_task = int(cur_task)
            for child in m.lora_A.values():
                if isinstance(child, OLoRALinearA):
                    child.cur_task = int(cur_task)
            for child in m.lora_B.values():
                if isinstance(child, OLoRALinearB):
                    child.cur_task = int(cur_task)


def apply_olora_expert_trainable_mask(
    model: nn.Module,
    cur_task: int,
    adapter_name: str = "default",
) -> None:
    """
    Only LoRA weights for slot ``cur_task`` train; other slots frozen.
    Call after ``mark_only_lora_as_trainable``.
    """
    # Match ``...loraA.i.mlp.weight`` or legacy ``...loraA.i.weight``
    pat = re.compile(rf"lora_[AB]\.{re.escape(adapter_name)}\.lora[AB]\.(\d+)\.(?:mlp\.)?weight$")
    for name, p in model.named_parameters():
        m = pat.search(name)
        if not m:
            continue
        tid = int(m.group(1))
        p.requires_grad = tid == int(cur_task)


def compute_olora_orthogonal_loss(
    model: nn.Module,
    cur_task: int,
    adapter_name: str = "default",
) -> torch.Tensor:
    """
    O-LoRA orthogonal regularizer: per layer prefix, sum ``sum |A_t @ A_i^T|`` for ``i < t`` over ``lora_A`` matrices.

    Uses L1 sum of absolute values (legacy LLaVA OLoRA script), not paper ``||A_i^T A_t||_F^2``.
    """
    if cur_task < 1:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        z = torch.zeros((), device=p.device, dtype=p.dtype)
        return z

    pattern = re.compile(rf"lora_A\.{re.escape(adapter_name)}\.loraA\.(\d+)\.(?:mlp\.)?weight$")
    current_by_prefix: Dict[str, torch.nn.Parameter] = {}
    prev_by_prefix: Dict[str, Dict[int, torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        m = pattern.search(name)
        if not m:
            continue
        tid = int(m.group(1))
        prefix = name[: m.start()]
        if tid == cur_task:
            current_by_prefix[prefix] = param
        elif 0 <= tid < cur_task:
            prev_by_prefix.setdefault(prefix, {})[tid] = param

    if not current_by_prefix:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device, dtype=p.dtype) if p is not None else torch.tensor(0.0)

    total: Optional[torch.Tensor] = None
    for prefix, a_cur in current_by_prefix.items():
        prevs = prev_by_prefix.get(prefix)
        if not prevs:
            continue
        for i in range(cur_task):
            a_prev = prevs.get(i)
            if a_prev is None:
                continue
            term = torch.abs(torch.mm(a_cur, a_prev.detach().T)).sum()
            total = term if total is None else total + term

    if total is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device, dtype=p.dtype) if p is not None else torch.tensor(0.0)
    return total
