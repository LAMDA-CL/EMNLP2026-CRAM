# -*- encoding: utf-8 -*-
"""
Standard MoE-LoRA (Mixture-of-Experts LoRA):

- **Router**: per-layer, per-token linear map to ``E`` logits, ``softmax`` -> gates ``g``.
- **Experts**: ``E`` LoRA branches; total rank ``r`` is split across experts (A: ``d→r/E``, B: ``r/E→d'``).
- **Output**: ``Δ = Σ_e g_e · B_e(A_e(x))``, scaled by ``lora_alpha/r`` and added to frozen base linear.

Independent of ``MOE_LORA_SAME``; ``task_embedding_dim`` is kept for legacy configs.
"""
import warnings
from dataclasses import dataclass, field

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


@dataclass
class MoELoRAConfig(LoraConfig):
    """MoE-LoRA: LoRA fields plus expert/task knobs (renamed from legacy SAME naming)."""

    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=8)
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_MOELORA


class MoELoRAModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _prepare_moelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_moelora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("MoELoRAModel supports only 1 adapter with bias.")

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
            "task_embedding_dim": lora_config.task_embedding_dim,
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
                    warnings.warn("fan_in_fan_out set to True for Linear layer; setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} not supported.")
            new_module = MoELoRALinear(
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
            raise ValueError("Cannot merge LoRA layers when model is loaded in 8-bit mode")
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


class MoELoRALayer(LoraLayer):
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
                        adapter_name: MoELoRALinearA(
                            self.in_features,
                            r,
                            self.expert_num,
                            self.cur_task,
                            self.training,
                            self.layer_id,
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: MoELoRALinearB(
                            r,
                            self.out_features,
                            self.expert_num,
                            self.cur_task,
                            self.training,
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


class MoELoRALinear(nn.Linear, MoELoRALayer):
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
        self.cur_task = kwargs.pop("cur_task", 0)
        self.layer_id = layer_id
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MoELoRALayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
            cur_task=self.cur_task,
            training=train_signal,
            layer_id=layer_id,
        )

        self.lora_router = nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)})

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
            router = self.lora_router[self.active_adapter]
            lead_shape = x_d.shape[:-1]
            flat = x_d.reshape(-1, x_d.size(-1)).to(dtype=router.weight.dtype)
            router_logits = router(flat)
            g = F.softmax(router_logits, dim=-1).view(*lead_shape, self.expert_num).to(dtype=x_d.dtype)
            lora_a_out = self.lora_A[self.active_adapter](x_d, g)
            lora_b_out = self.lora_B[self.active_adapter](lora_a_out, g)
            result = result + lora_b_out * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result


class MoELoRALinearA(nn.Module):
    """Expert LoRA A banks; ``routing_weights`` aligns with ``x`` except last dim, shape ``(..., E)``."""

    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        r_per = out_features // expert_num
        if r_per * expert_num != out_features:
            raise ValueError(f"LoRA rank r={out_features} must be divisible by expert_num={expert_num}.")
        self.loraA = nn.ModuleList([MoELoRAExpert(in_features, r_per) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        # routing_weights: (..., E), x: (..., in_features)
        g = routing_weights
        out = self.loraA[0](x) * g[..., 0:1]
        for i in range(1, self.expert_num):
            out = out + self.loraA[i](x) * g[..., i : i + 1]
        return out


class MoELoRALinearB(nn.Module):
    """Expert LoRA B banks; ``routing_weights`` shape ``(..., E)``."""

    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        r_per = in_features // expert_num
        if r_per * expert_num != in_features:
            raise ValueError(f"LoRA rank r={in_features} must be divisible by expert_num={expert_num}.")
        self.loraB = nn.ModuleList([MoELoRAExpert(r_per, out_features) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        g = routing_weights
        out = self.loraB[0](x) * g[..., 0:1]
        for i in range(1, self.expert_num):
            out = out + self.loraB[i](x) * g[..., i : i + 1]
        return out


class MoELoRAExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)
