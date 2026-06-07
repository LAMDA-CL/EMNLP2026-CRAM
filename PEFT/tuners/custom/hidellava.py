# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
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

from ...import_utils import is_bnb_4bit_available, is_bnb_available, linear8bitlt_init_kwargs

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class HiDeMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_HiDe`]
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    cur_task: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_HiDe


class HiDeMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        self.predicted_task_id=-1
        print(self.model.training)

    #added by me
    def set_predicted_task_id(self, task_id: int):
        self.predicted_task_id = task_id
        # Propagate to every HiDe MoE LoRA layer
        for module in self.model.modules():
            if isinstance(module, HiDeMOELoraLinear):
                module.predicted_task_id = task_id

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_hidemoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
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
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

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
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": getattr(target.state, "memory_efficient_backward", False),
                    "threshold": target.state.threshold,
                    "index": target.index,
                    "layer": layer,
                }
            )
            try:
                from config.backbone.registry import get_last_lora_block_index

                eightbit_kwargs["last_layer_idx"] = get_last_lora_block_index()
            except ImportError:
                eightbit_kwargs["last_layer_idx"] = None
            new_module = HiDeMOELoraLinear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, train_signal=training, **eightbit_kwargs
            )
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
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            try:
                from config.backbone.registry import get_last_lora_block_index

                last_layer_idx = get_last_lora_block_index()
            except ImportError:
                last_layer_idx = None
            new_module = HiDeMOELoraLinear(
                adapter_name,
                in_features,
                out_features,
                bias=bias,
                train_signal=training,
                layer=layer,
                last_layer_idx=last_layer_idx,
                **kwargs,
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_hidemoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

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
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
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
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class HiDeMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int, cur_task: int, training: bool, layer: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.training = training
        self.layer = layer

    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: HiDeMOELinearA(self.in_features, r, self.expert_num, self.cur_task, self.training, self.layer)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: HiDeMOELinearB(r, self.out_features, self.expert_num, self.cur_task, self.training, self.layer)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class HiDeMOELoraLinear(nn.Linear, HiDeMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        train_signal: bool = False,
        layer: int = 0,
        last_layer_idx: Optional[int] = None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.cur_task = kwargs.pop("cur_task", True)
        last_layer_idx = kwargs.pop("last_layer_idx", last_layer_idx)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        HiDeMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num,
                               cur_task=self.cur_task,
                               training=train_signal,
                               layer=layer,
                               )

        self.layer = layer
        self.last_layer_idx = last_layer_idx
        self.training = train_signal
        self.predicted_task_id=-1

        # init the Gate network
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def _parse_layer_index(self) -> Optional[int]:
        try:
            return int(self.layer)
        except (TypeError, ValueError):
            return None

    def _inference_use_predicted_expert(self) -> bool:
        """Last transformer block uses predicted_task_id for expert selection; other layers fuse experts."""
        if self.last_layer_idx is None:
            return True
        idx = self._parse_layer_index()
        if idx is None:
            return True
        return idx == self.last_layer_idx

    def _lora_delta_one_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        adapter = self.active_adapter
        a_out = self.lora_A[adapter](x, expert_idx)
        return self.lora_B[adapter](a_out, expert_idx)

    def _lora_delta_fused_experts(self, x: torch.Tensor) -> torch.Tensor:
        total = self._lora_delta_one_expert(x, 0)
        for k in range(1, self.expert_num):
            total = total + self._lora_delta_one_expert(x, k)
        return total

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

        # No LoRA adapter registered (base linear only)
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # Adapters disabled
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            if self.training:
                task_id = self.cur_task
                lora_a_output = self.lora_A[self.active_adapter](x, task_id)
                lora_b_output = self.lora_B[self.active_adapter](lora_a_output, task_id)
                result += lora_b_output * self.scaling[self.active_adapter]
            else:
                # Inference: last layer picks one expert via predicted_task_id; others sum all expert deltas (fuse)
                if self._inference_use_predicted_expert():
                    if self.predicted_task_id == -1:
                        raise RuntimeError(
                            f"HiDeMOELoraLinear layer={self.layer!r}: inference on last layer requires "
                            f"predicted_task_id >= 0, got {self.predicted_task_id}"
                        )
                    task_id = self.predicted_task_id
                    lora_a_output = self.lora_A[self.active_adapter](x, task_id)
                    lora_b_output = self.lora_B[self.active_adapter](lora_a_output, task_id)
                    result += lora_b_output * self.scaling[self.active_adapter]
                else:
                    lora_b_output = self._lora_delta_fused_experts(x)
                    result += lora_b_output * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


# HiDeMOELinearA/B wrap ModuleList of per-task expert LoRA blocks (not plain 2D LoRA matrices).
# expert_num submodules are created up front (one slot per task).
class HiDeMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer) -> None:

        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])
        self.training = training
        self.layer = layer

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(HiDeMOEExpert(self.in_features, self.r))

    def forward(self, x, task_id):
        #print(f"HiDeMOELinearA layer {self.layer} using task_id {task_id}")
        return self.loraA[task_id](x)

class HiDeMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer) -> None:

        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])
        self.training = training
        self.layer = layer

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(HiDeMOEExpert(self.r, self.out_features))

    def forward(self, x, task_id):
        return self.loraB[task_id](x)

class HiDeMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    
    def forward(self, x):
        return self.mlp(x)


if is_bnb_available():

    _HIDE_MOE_LINEAR_SHARED = (
        "merge",
        "unmerge",
        "_parse_layer_index",
        "_inference_use_predicted_expert",
        "_lora_delta_one_expert",
        "_lora_delta_fused_experts",
    )

    class HiDeMOELoraLinear8bitLt(bnb.nn.Linear8bitLt, HiDeMOELoraLayer):
        """8-bit HiDe MoE-LoRA with CLIP-routed predicted_task_id at inference."""

        def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            train_signal: bool = False,
            layer: int = 0,
            last_layer_idx: Optional[int] = None,
            **kwargs,
        ):
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.expert_num = kwargs.pop("expert_num", 8)
            kwargs.pop("task_embedding_dim", None)
            self.cur_task = kwargs.pop("cur_task", 0)
            kwargs.pop("fan_in_fan_out", None)
            bias = kwargs.pop("bias", True)

            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                **linear8bitlt_init_kwargs(bias=bias, **kwargs),
            )
            HiDeMOELoraLayer.__init__(
                self,
                in_features=in_features,
                out_features=out_features,
                expert_num=self.expert_num,
                cur_task=self.cur_task,
                training=train_signal,
                layer=layer,
            )
            self.layer = layer
            self.last_layer_idx = last_layer_idx
            self.training = train_signal
            self.predicted_task_id = -1
            self.lora_router = nn.ModuleDict(
                {adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}
            )
            self.weight.requires_grad = False
            self.fan_in_fan_out = False
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor, **kwargs):
            previous_dtype = x.dtype
            if self.active_adapter not in self.lora_A.keys():
                return bnb.nn.Linear8bitLt.forward(self, x)
            if self.disable_adapters:
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
                return bnb.nn.Linear8bitLt.forward(self, x)
            if self.r[self.active_adapter] > 0:
                result = bnb.nn.Linear8bitLt.forward(self, x)
                result = result.clone()
                x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
                if self.training:
                    task_id = self.cur_task
                    lora_a_output = self.lora_A[self.active_adapter](x, task_id)
                    lora_b_output = self.lora_B[self.active_adapter](lora_a_output, task_id)
                    result += lora_b_output * self.scaling[self.active_adapter]
                elif self._inference_use_predicted_expert():
                    if self.predicted_task_id == -1:
                        raise RuntimeError(
                            f"HiDeMOELoraLinear8bitLt layer={self.layer!r}: inference on last layer requires "
                            f"predicted_task_id >= 0, got {self.predicted_task_id}"
                        )
                    task_id = self.predicted_task_id
                    lora_a_output = self.lora_A[self.active_adapter](x, task_id)
                    lora_b_output = self.lora_B[self.active_adapter](lora_a_output, task_id)
                    result += lora_b_output * self.scaling[self.active_adapter]
                else:
                    lora_b_output = self._lora_delta_fused_experts(x)
                    result += lora_b_output * self.scaling[self.active_adapter]
                return result.to(previous_dtype)
            return bnb.nn.Linear8bitLt.forward(self, x)

    for _name in _HIDE_MOE_LINEAR_SHARED:
        setattr(HiDeMOELoraLinear8bitLt, _name, getattr(HiDeMOELoraLinear, _name))

