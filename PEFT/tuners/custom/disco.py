# -*- encoding: utf-8 -*-
"""
DISCO (CoIN) MoE-LoRA PEFT Tuner

Key design:
  - Training: single expert activated via `lora_id`
  - Inference: all experts concatenated, weighted by `lora_AB` diagonal mask
    whose values are set from cosine-similarity routing signals (`mask_signal`).
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

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

from ...import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DiscoMOELoraConfig(LoraConfig):
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=8)
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_DisCo


class DiscoMOELoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_discomodelora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "DiscoMOELoraModel supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

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
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, self.model.training)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target, training):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "cur_task": lora_config.cur_task,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update({
                "has_fp16_weights": target.state.has_fp16_weights,
                "memory_efficient_backward": target.state.memory_efficient_backward,
                "threshold": target.state.threshold,
                "index": target.index,
            })
            new_module = Linear8bitLt(adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update({
                "compute_dtype": target.compute_dtype,
                "compress_statistics": target.weight.compress_statistics,
                "quant_type": target.weight.quant_type,
            })
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
                    warnings.warn("fan_in_fan_out is set to True but target is Linear. Setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} is not supported.")
            new_module = DiscoMOELoraLinear(
                adapter_name, in_features, out_features,
                bias=bias, train_signal=training, **kwargs
            )
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_discomodelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config


# ---------------------------------------------------------------------------
#  Layer classes
# ---------------------------------------------------------------------------

class DiscoMOELoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, expert_num: int, cur_task: int, training: bool):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.training = training

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_A.update(nn.ModuleDict({
                adapter_name: DiscoMOELinearA(self.in_features, r, self.expert_num, self.cur_task, self.training)
            }))
            self.lora_B.update(nn.ModuleDict({
                adapter_name: DiscoMOELinearB(r, self.out_features, self.expert_num, self.cur_task, self.training)
            }))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class DiscoMOELoraLinear(nn.Linear, DiscoMOELoraLayer):
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
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", 8)
        self.te_dim = kwargs.pop("task_embedding_dim", 64)
        self.cur_task = kwargs.pop("cur_task", 0)
        self.lora_id = self.cur_task
        self.mask_signal = [1.0] + [0.0] * (self.expert_num - 1)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DiscoMOELoraLayer.__init__(
            self, in_features=in_features, out_features=out_features,
            expert_num=self.expert_num, cur_task=self.cur_task, training=train_signal,
        )
        self.training = train_signal

        # lora_AB: diagonal mask for soft expert routing at inference
        self.lora_AB = nn.ModuleDict({})
        self.lora_AB.update(nn.ModuleDict({adapter_name: nn.Linear(r, r, bias=False)}))
        self.lora_AB[adapter_name].weight.data.zero_()
        self.diag_size = r // self.expert_num

        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            if self.training:
                lora_a_output = self.lora_A[self.active_adapter].loraA[self.lora_id](
                    self.lora_dropout[self.active_adapter](x)
                )
                lora_b_output = self.lora_B[self.active_adapter].loraB[self.lora_id](lora_a_output)
                result += lora_b_output * self.scaling[self.active_adapter]
            else:
                # --- Inference: soft routing via lora_AB diagonal mask ---
                # Write mask_signal (cosine similarities) into lora_AB diagonal blocks
                for i, flag in enumerate(self.mask_signal):
                    if flag != 0:
                        start = i * self.diag_size
                        end = start + self.diag_size
                        if end > start and self.diag_size > 0:
                            self.lora_AB[self.active_adapter].weight.data[start:end, start:end] = (
                                torch.eye(self.diag_size, device=x.device, dtype=x.dtype) * flag
                            )

                lora_a_output = self.lora_A[self.active_adapter](x)        # all experts concat
                lora_ab_output = self.lora_AB[self.active_adapter](lora_a_output)  # diagonal weight
                lora_b_output = self.lora_B[self.active_adapter](lora_ab_output)   # all experts concat
                result += lora_b_output * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result


class DiscoMOELinearA(nn.Module):
    """MoE-based LoRA A block: ModuleList of experts (train: single; infer: concatenated)."""
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])
        self.training = training
        if self.out_features % self.expert_num != 0:
            raise ValueError(
                f"DiscoMOELinearA: r={self.out_features} not divisible by expert_num={self.expert_num}. "
                f"Adjust to a multiple of {self.expert_num}."
            )
        self.r = self.out_features // self.expert_num
        for _ in range(self.expert_num):
            self.loraA.append(DiscoMOEExpert(self.in_features, self.r))

    def forward(self, x):
        if self.training:
            output = self.loraA[self.cur_task](x)
            return output
        else:
            total_out = self.expert_num * self.r
            temp_mlp = nn.Linear(self.in_features, total_out, bias=False).to(x.device)
            concatenated_weight = torch.cat([e.mlp.weight for e in self.loraA], dim=0)
            with torch.no_grad():
                temp_mlp.weight.copy_(concatenated_weight)
            return temp_mlp(x)


class DiscoMOELinearB(nn.Module):
    """MoE-based LoRA B block: ModuleList of experts (train: single; infer: concatenated)."""
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])
        self.training = training
        if self.in_features % self.expert_num != 0:
            raise ValueError(
                f"DiscoMOELinearB: r={self.in_features} not divisible by expert_num={self.expert_num}. "
                f"Adjust to a multiple of {self.expert_num}."
            )
        self.r = self.in_features // self.expert_num
        for _ in range(self.expert_num):
            self.loraB.append(DiscoMOEExpert(self.r, self.out_features))

    def forward(self, x):
        if self.training:
            output = self.loraB[self.cur_task](x)
            return output
        else:
            total_in = self.expert_num * self.r
            temp_mlp = nn.Linear(total_in, self.out_features, bias=False).to(x.device)
            concatenated_weight = torch.cat([e.mlp.weight for e in self.loraB], dim=1)
            with torch.no_grad():
                temp_mlp.weight.copy_(concatenated_weight)
            return temp_mlp(x)


class DiscoMOEExpert(nn.Module):
    """Single LoRA expert (one A or B matrix)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)
