# SMoLoRA: Separable Mixture of LoRA (ported from SMoLoRA/PEFT_SMoLoRA, adapted to this repo's PEFT/LoraModel APIs)
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

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
class SMoLoraConfig(LoraConfig):
    """
    SMoLoRA hyperparameters.

    ``expert_num`` is the **total** VU + IF experts (must be even): first ``expert_num/2`` slots are VU-only
    (``lora_vu_gate`` top-1 among ``expert_num/2``), last ``expert_num/2`` are IF-only (``lora_ins_gate``).
    For **4 VU + 4 IF** routes use ``expert_num=8``.
    """

    expert_num: int = field(default=8)
    ins_type: int = field(default=0)
    ins_emb_dim: int = field(default=768)
    # Offline Sentence-BERT matrix ``[num_instructions, D]`` (e.g. ``ins_emb_single.pkl`` from ``ins_gen.py``);
    # when set, ``SMoLoraLinear`` picks row ``ins_type`` and broadcasts (paper-aligned).
    ins_emb: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.expert_num < 2 or self.expert_num % 2 != 0:
            raise ValueError("SMoLoRA `expert_num` must be a positive even integer.")
        if self.r <= 0 or self.r % self.expert_num != 0:
            raise ValueError(f"SMoLoRA requires LoRA rank `r` ({self.r}) divisible by `expert_num` ({self.expert_num}).")
        self.peft_type = PeftType.SMOLORA


# CLIP towers use Linear layers named like q_proj; do not wrap those or CLIP forward breaks before integration writes feats.
_SMOLORA_REPLACE_SKIP_SUBSTR = (
    "vision_tower",
    "text_tower",
    "mm_projector",
    "vision_resampler",
)

# Log IF-side (lora_ins_gate) top-1 expert only on these decoder block indices (local idx 0..expert_num/2-1)
_IF_ROUTE_LOG_LAYERS = frozenset({10, 20, 30})


def _parse_llama_layer_index(module_key: Optional[str]) -> Optional[int]:
    if not module_key:
        return None
    m = re.search(r"\.layers\.(\d+)\.", module_key)
    if m:
        return int(m.group(1))
    m = re.search(r"layers\.(\d+)", module_key)
    if m:
        return int(m.group(1))
    return None


class SMoLoraModel(LoraModel):
    """Wraps the base model and replaces target Linear layers with `SMoLoraLinear`."""

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_smolora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "SMoLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none'."
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
            if any(s in key for s in _SMOLORA_REPLACE_SKIP_SUBSTR):
                continue
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, _layer = _get_submodules(self.model, key)

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
                new_module = self._create_new_module(lora_config, adapter_name, target, module_key=key)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target, module_key=None):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
            "ins_type": lora_config.ins_type,
            "ins_emb_dim": lora_config.ins_emb_dim,
            "ins_emb": getattr(lora_config, "ins_emb", None),
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
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
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
            line_kwargs = dict(kwargs)
            if module_key is not None:
                li = _parse_llama_layer_index(module_key)
                if li is not None:
                    line_kwargs["layer_idx"] = li
                line_kwargs["module_key"] = module_key
            new_module = SMoLoraLinear(adapter_name, in_features, out_features, bias=bias, **line_kwargs)

        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_smolora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
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
            except AttributeError:
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
                    if merge:
                        target.merge()
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


class SMoLoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, expert_num: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: SMoLoraLinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: SMoLoraLinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class SMoLoraLinear(nn.Linear, SMoLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        expert_num = int(kwargs.pop("expert_num", 8))
        ins_type = int(kwargs.pop("ins_type", 0))
        ins_emb = kwargs.pop("ins_emb", None)
        layer_idx = kwargs.pop("layer_idx", None)
        module_key = kwargs.pop("module_key", None)
        ins_emb_dim = int(kwargs.pop("ins_emb_dim", 768))
        kwargs.pop("is_target_conv_1d_layer", None)
        kwargs.pop("fan_in_fan_out", None)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SMoLoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=expert_num)

        self.layer_idx: Optional[int] = layer_idx
        self._module_key: Optional[str] = module_key
        self.ins_type = ins_type
        self.ins_emb_dim = ins_emb_dim
        # Nested list [T, D] like ins_emb_single.pkl; row ins_type (paper-aligned).
        self.ins_emb = ins_emb

        self.lora_vu_gate = nn.ModuleDict(
            {adapter_name: nn.Linear(self.in_features, self.expert_num // 2, bias=False)}
        )
        self.lora_ins_gate = nn.ModuleDict(
            {adapter_name: nn.Linear(self.ins_emb_dim, self.expert_num // 2, bias=False)}
        )
        self.lora_fc_A = nn.ModuleDict({adapter_name: nn.Linear(self.out_features, 1, bias=False)})
        self.lora_fc_B = nn.ModuleDict({adapter_name: nn.Linear(self.out_features, 1, bias=False)})

        self.weight.requires_grad = False
        if self.ins_type > 0:
            for param in self.lora_vu_gate.parameters():
                param.requires_grad = False
        else:
            for param in self.lora_vu_gate.parameters():
                param.requires_grad = True
        if self.ins_type > 0:
            for param in self.lora_ins_gate.parameters():
                param.requires_grad = False
        else:
            for param in self.lora_ins_gate.parameters():
                param.requires_grad = True

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

    def _log_if_route_top1(self, max_indices_if: torch.Tensor) -> None:
        """Log IF-side (lora_ins_gate) top-1 expert index; local slot in [0, expert_num/2)."""
        if self.layer_idx is None or self.layer_idx not in _IF_ROUTE_LOG_LAYERS:
            return
        tail = self._module_key.split(".")[-1] if self._module_key else "?"
            # One log line per layer: use q_proj as representative for the block's IF route
        if tail != "q_proj":
            return
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass
        local = int(max_indices_if.reshape(-1)[0].item())
        print(f"[SMoLoRA-IF-route] layer={self.layer_idx} top1_IF_expert={local}", flush=True)

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
            x = x.to(self.lora_A[self.active_adapter].loraA[0].mlp.weight.dtype)

            # Paper: offline ins_emb row then broadcast; else CLIP features [B, ins_emb_dim] from integration
            b = x.shape[0]
            if self.ins_emb is not None:
                mat = torch.as_tensor(self.ins_emb, dtype=x.dtype, device=x.device)
                if mat.dim() == 1:
                    mat = mat.unsqueeze(0)
                ti = max(0, min(int(self.ins_type), mat.shape[0] - 1))
                vec = mat[ti]
                ins_clip = vec.unsqueeze(0).expand(b, -1)
            else:
                runtime = getattr(self, "_runtime_instruction_feat", None)
                if runtime is None:
                    raise RuntimeError(
                        "SMoLoRA needs instruction features: set `ins_emb` in `SMoLoraConfig` (offline .pkl), "
                        "or use `method.custom.smolora` with `text_tower` / `clip_tokenizer` to fill `_runtime_instruction_feat`."
                    )
                ins_clip = runtime.to(device=x.device, dtype=x.dtype)
                if ins_clip.dim() == 1:
                    ins_clip = ins_clip.unsqueeze(0)
                if ins_clip.shape[-1] != self.ins_emb_dim:
                    raise RuntimeError(
                        f"runtime instruction feat dim {ins_clip.shape[-1]} != lora_ins_gate.in_features {self.ins_emb_dim}"
                    )
                if ins_clip.shape[0] == 1 and b > 1:
                    ins_clip = ins_clip.expand(b, -1)
                elif ins_clip.shape[0] != b:
                    raise RuntimeError(
                        f"runtime instruction batch {ins_clip.shape[0]} != hidden batch {b}"
                    )

            self.lora_vu_gate = self.lora_vu_gate.to(x.device)
            self.lora_ins_gate = self.lora_ins_gate.to(x.device)

            x_emb = torch.mean(x, dim=1, keepdim=True)
            vu_router = self.lora_vu_gate[self.active_adapter](x_emb)
            top1_vu_router = torch.zeros_like(vu_router)
            max_values, max_indices = torch.max(vu_router, dim=-1, keepdim=True)
            top1_vu_router.scatter_(-1, max_indices, 1.0)

            if_router = self.lora_ins_gate[self.active_adapter](ins_clip)
            top1_if_router = torch.zeros_like(if_router)
            max_values_if, max_indices_if = torch.max(if_router, dim=-1, keepdim=True)
            top1_if_router.scatter_(-1, max_indices_if, 1.0)
            self._log_if_route_top1(max_indices_if)
            top1_if_router = top1_if_router.unsqueeze(1)

            final_router = torch.cat((top1_vu_router, top1_if_router), dim=-1)
            self.final_router = final_router

            vu_result = 0.0
            if_result = 0.0
            for i in range(self.expert_num // 2):
                vu_result = vu_result + (
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * final_router[:, :, i].unsqueeze(-1)
                )

            for i in range(self.expert_num // 2, self.expert_num):
                if_result = if_result + (
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * final_router[:, :, i].unsqueeze(-1)
                )

            score_A = self.lora_fc_A[self.active_adapter](vu_result)
            score_B = self.lora_fc_B[self.active_adapter](if_result)
            scores = torch.cat([score_A, score_B], dim=-1)
            attention_weights = F.softmax(scores, dim=-1)
            self.attention_weights = attention_weights

            alpha_A = attention_weights[:, :, 0].unsqueeze(-1)
            alpha_B = attention_weights[:, :, 1].unsqueeze(-1)
            A_weighted = alpha_A * vu_result
            B_weighted = alpha_B * if_result
            result = result + (A_weighted + B_weighted)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


class SMoLoraLinearA(nn.Module):
    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])
        if self.out_features % self.expert_num != 0:
            raise ValueError("LoRA rank r must be divisible by expert_num for SMoLoraLinearA.")
        self.r = self.out_features // self.expert_num
        for _ in range(self.expert_num):
            self.loraA.append(SMoExpert(self.in_features, self.r))

    def forward(self, x):
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))
        return outputs


class SMoLoraLinearB(nn.Module):
    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])
        if self.in_features % self.expert_num != 0:
            raise ValueError("LoRA rank r must be divisible by expert_num for SMoLoraLinearB.")
        self.r = self.in_features // self.expert_num
        for _ in range(self.expert_num):
            self.loraB.append(SMoExpert(self.r, self.out_features))

    def forward(self, x):
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))
        return outputs


class SMoExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)
