# -*- encoding: utf-8 -*-
import inspect
import warnings
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from typing import Optional
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
    Linear4bit,
    Embedding,
    Conv2d,
)
from ...import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


# PyTorch non-reentrant checkpoint runs the user forward from
# unpack_hook -> _run_fn_with_dynamo_disabled -> recompute_fn (see torch/utils/checkpoint.py).
_CHECKPOINT_FRAME_NAMES = frozenset({"recompute_fn", "_run_fn_with_dynamo_disabled"})


def _same_inside_activation_checkpoint_recompute() -> bool:
    """True when executing the activation-checkpoint *recompute* forward (not the first pass)."""
    frame = inspect.currentframe()
    depth = 0
    try:
        f = frame
        while f is not None and depth < 256:
            depth += 1
            co = f.f_code
            if co.co_name in _CHECKPOINT_FRAME_NAMES:
                path = co.co_filename.replace("\\", "/")
                if "torch/utils/checkpoint" in path:
                    return True
            f = f.f_back
    finally:
        del frame
    return False


@dataclass
class SAMEConfig(LoraConfig):
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=8)
    cur_task: int = field(default=0)
    tau_score: float = field(default=0.1)
    curvature_mu: float = field(default=0.9)
    window_size: int = field(default=3)
    max_components: int = field(default=64)
    # Spectral / covariance principal directions: keep dirs until cumulative energy reaches this ratio (router + curvature hooks)
    cumulative_energy_ratio: float = field(default=0.9)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_SAME


class SAMEModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_SAME_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("MMOELoraModel supports only 1 adapter with bias.")

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
            "tau_score": lora_config.tau_score,
            "curvature_mu": lora_config.curvature_mu,
            "window_size": lora_config.window_size,
            "max_components": lora_config.max_components,
            "cumulative_energy_ratio": lora_config.cumulative_energy_ratio,
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
            new_module = SAMELinear8bitLt(
                adapter_name,
                target.in_features,
                target.out_features,
                bias=bias,
                train_signal=training,
                **eightbit_kwargs,
            )
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
                    warnings.warn("fan_in_fan_out set to True for Linear layer; setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape)
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} not supported.")
            new_module = SAMELinear(
                adapter_name, in_features, out_features, bias=bias, 
                train_signal=training, **kwargs
            )
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_SAME_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

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
                        target.in_channels, target.out_channels,
                        kernel_size=target.kernel_size, stride=target.stride,
                        padding=target.padding, dilation=target.dilation,
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


class SAMELayer(LoraLayer):
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
            self.lora_A.update(nn.ModuleDict({adapter_name: SAMELinearA(
                self.in_features, r, self.expert_num, self.cur_task, self.training, self.layer_id)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: SAMELinearB(
                r, self.out_features, self.expert_num, self.cur_task, self.training, self.layer_id)}))
            self.scaling[adapter_name] = lora_alpha / r

            # Explicitly mark all expert parameters as trainable
            for expert_id in range(self.expert_num):
                for param in self.lora_A[adapter_name].loraA[expert_id].parameters():
                    param.requires_grad_(True)
                for param in self.lora_B[adapter_name].loraB[expert_id].parameters():
                    param.requires_grad_(True)


        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)





_SAME_COV_PREV_HOOK_WARNED: bool = False


def _same_pop_layer_kwargs(kwargs: dict) -> dict:
    return {
        "init_lora_weights": kwargs.pop("init_lora_weights", True),
        "expert_num": kwargs.pop("expert_num", 8),
        "te_dim": kwargs.pop("task_embedding_dim", 64),
        "cur_task": kwargs.pop("cur_task", 0),
        "max_components": int(kwargs.pop("max_components", 64)),
        "window_size": int(kwargs.pop("window_size", 3)),
        "curvature_mu": float(kwargs.pop("curvature_mu", 0.9)),
        "tau_score": float(kwargs.pop("tau_score", 0.1)),
        "cumulative_energy_ratio": float(kwargs.pop("cumulative_energy_ratio", 0.9)),
    }


def _same_init_runtime_state(
    layer,
    adapter_name: str,
    in_features: int,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    init_lora_weights: bool,
    train_signal: bool,
) -> None:
    layer.router = torch.ones(layer.expert_num)
    layer.lora_router = nn.ModuleDict({adapter_name: SAMEExpert(layer.in_features, layer.expert_num)})
    layer.weight.requires_grad = False
    layer.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
    layer.active_adapter = adapter_name
    layer.register_buffer(f"cov_U_{adapter_name}", torch.zeros(in_features, layer.max_components))
    layer.register_buffer(f"cov_S_{adapter_name}", torch.zeros(layer.max_components))
    layer.register_buffer(f"cov_alpha_{adapter_name}", torch.tensor(0.0))
    layer.register_buffer(f"cov_U_prev_{adapter_name}", torch.zeros(in_features, layer.max_components))
    layer.register_buffer(f"cov_S_prev_{adapter_name}", torch.zeros(layer.max_components))
    layer.register_buffer(f"cov_prev_valid_{adapter_name}", torch.tensor(False))
    layer.register_buffer(f"utilization_{adapter_name}", torch.ones(layer.expert_num) / layer.expert_num)
    layer.register_buffer(f"importance_{adapter_name}", torch.ones(layer.expert_num) / layer.expert_num)
    layer.register_buffer(f"expert_masks_{adapter_name}", torch.ones(layer.expert_num))
    layer.current_step = 0
    layer.start_step = 10
    layer.all_frozen = False
    layer.router_scaling_factor = 0.2
    layer._register_same_hooks(adapter_name)


def _same_forward_with_routing(layer, x: torch.Tensor, base_result: torch.Tensor) -> torch.Tensor:
    previous_dtype = x.dtype
    if layer.r[layer.active_adapter] > 0:
        x = x.to(layer.lora_A[layer.active_adapter].loraA[0].weight.dtype)
        if layer.training:
            x_flat = x.view(-1, x.size(-1))
            if not _same_inside_activation_checkpoint_recompute():
                layer._update_router_covariance(x_flat, layer.active_adapter)
            router_logits = layer.lora_router[layer.active_adapter](x.view(-1, x.size(-1)))
            g_phi = F.softmax(router_logits, dim=-1).to(dtype=x.dtype)
            g_phi_mean = g_phi.mean(dim=0).to(dtype=x.dtype)
            g_phi_mean = layer._get_masked_routing(g_phi_mean)
            final_routing = g_phi_mean.to(dtype=x.dtype)
        else:
            router_logits = layer.lora_router[layer.active_adapter](x.view(-1, x.size(-1))) / layer.expert_num
            g_phi = F.softmax(router_logits, dim=-1)
            g_phi_mean = g_phi.mean(dim=0)
            final_routing = layer._apply_topk_sparsification(g_phi_mean, layer.router, k=2).to(dtype=x.dtype)

        final_routing = final_routing.to(layer.lora_A[layer.active_adapter].loraA[0].weight.dtype)
        lora_a_output = layer.lora_A[layer.active_adapter](x, final_routing)
        lora_b_output = layer.lora_B[layer.active_adapter](lora_a_output, final_routing)
        result = base_result + lora_b_output * layer.scaling[layer.active_adapter]
    else:
        result = base_result

    result = result.to(previous_dtype)
    if layer.training and layer.r[layer.active_adapter] > 0 and not _same_inside_activation_checkpoint_recompute():
        layer.current_step += 1
        with torch.no_grad():
            layer._update_activation_metrics(g_phi, x.view(-1, x.size(-1)))
            if not layer.all_frozen:
                layer._apply_adaptive_freezing()
    return result


class SAMELinear(nn.Linear, SAMELayer):
    """LoRA SAME linear; registers curvature hooks on expert A weights."""

    _hook_skip_cov_prev_warned: bool = False

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
        cfg = _same_pop_layer_kwargs(kwargs)
        init_lora_weights = cfg["init_lora_weights"]
        self.expert_num = cfg["expert_num"]
        self.te_dim = cfg["te_dim"]
        self.cur_task = cfg["cur_task"]
        self.layer_id = layer_id
        self.max_components = cfg["max_components"]
        self.window_size = cfg["window_size"]
        self.curvature_mu = cfg["curvature_mu"]
        self.tau_score = cfg["tau_score"]
        self.cumulative_energy_ratio = cfg["cumulative_energy_ratio"]
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SAMELayer.__init__(
            self, in_features=in_features, out_features=out_features,
            expert_num=self.expert_num, cur_task=self.cur_task,
            training=train_signal, layer_id=layer_id,
        )

        self.training_signal = train_signal
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        _same_init_runtime_state(
            self, adapter_name, in_features, r, lora_alpha, lora_dropout, init_lora_weights, train_signal
        )

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


    def _register_same_hooks(self, adapter_name):
        router_weight = self.lora_router[adapter_name].weight
        router_weight.register_hook(lambda grad: self._spectral_aware_router_hook(grad, adapter_name))


        for name in ['loraA']:
            module_dict = getattr(self, f'lora_{name[4]}')
            if adapter_name not in module_dict:
                continue
            experts = module_dict[adapter_name].loraA if name == 'loraA' else module_dict[adapter_name].loraB
            for expert_id in range(len(experts)):
                weight = experts[expert_id].mlp.weight
                if not weight.requires_grad:
                    assert 0, (
                        f"SAME assert(0): layer={self.layer_id} expert={expert_id} {name}.weight "
                        f"requires_grad=False — curvature hook cannot run; check DeepSpeed/PEFT freeze."
                    )
                hook_fn = self._make_curvature_hook(name, expert_id, adapter_name)
                weight.register_hook(hook_fn)


    def _spectral_aware_router_hook(self, grad, adapter):
        if not self.training or self.current_step < self.start_step:
            return grad


        U = getattr(self, f"cov_U_{adapter}")
        S = getattr(self, f"cov_S_{adapter}")
        # Energy threshold 0.95
        energy = S ** 2
        total_energy = energy.sum()
        
        if total_energy == 0:
            assert(0)
            return grad
        
        # Cumulative energy ratio along singular values
        cumsum = torch.cumsum(energy, dim=0)
        ratio = cumsum / total_energy
        
        # Keep principal directions until cumulative_energy_ratio
        thr = self.cumulative_energy_ratio
        k = (ratio <= thr).sum().item() + 1
        k = min(k, len(S))  # clamp
        
        if k == 0:
            assert(0)
            return grad

        V_parallel = U[:, :k] 
        grad_parallel = grad @ V_parallel @ V_parallel.t()

        # Smooth singular values in a sliding window
        S_smooth = torch.zeros_like(S[:k])
        for i in range(k):
            start = max(0, i - self.window_size + 1)
            S_smooth[i] = S[start:i+1].mean()
        

        scaling = 1.0 / (S_smooth + 1e-6)
        scaling = torch.clamp(scaling / scaling.max(), min=1.0 - self.router_scaling_factor,max=1.0 + self.router_scaling_factor)

        grad_parallel_scaled = (grad_parallel @ V_parallel) * scaling.unsqueeze(0)
        grad_parallel_scaled = grad_parallel_scaled @ V_parallel.t()


        grad_perp = grad - grad_parallel
        return grad_parallel_scaled + grad_perp
    
    def _make_curvature_hook(self, name, expert_id, adapter):
        def hook(grad):
            if not self.training or self.cur_task == 0:
                return grad
            cov_prev_valid = getattr(self, f"cov_prev_valid_{adapter}", None)
            if cov_prev_valid is None:
                assert(0)
            if isinstance(cov_prev_valid, torch.Tensor):
                if cov_prev_valid.numel() != 1:
                   assert(0)
                if not bool(cov_prev_valid.item()):
                    U_prev = getattr(self, f"cov_U_prev_{adapter}", None)
                    S_prev = getattr(self, f"cov_S_prev_{adapter}", None)
                    if U_prev is not None and S_prev is not None:
                        e = (S_prev ** 2).sum()
                        if bool((e > 1e-12).item()) or bool((U_prev.abs().sum() > 1e-6).item()):
                            assert 0, (
                                f"SAME assert(0): cov_prev_valid_{adapter} is False but cov_*_prev "
                                f"carry non-trivial data (layer={self.layer_id}, expert={expert_id}); "
                                f"checkpoint / buffer state inconsistent."
                            )
                    global _SAME_COV_PREV_HOOK_WARNED
                    if not _SAME_COV_PREV_HOOK_WARNED:
                        _rk = 0
                        try:
                            import torch.distributed as dist

                            if dist.is_available() and dist.is_initialized():
                                _rk = int(dist.get_rank())
                        except Exception:
                            pass
                        if _rk == 0:
                            _SAME_COV_PREV_HOOK_WARNED = True
                            print(
                                "[SAME_HOOK] Skipping curvature block"
                                f"because cov_prev_valid_{adapter}=False — e.g. layer={self.layer_id} "
                                f"expert={expert_id}. Most hooks return here until Task0 snapshot + load restore True.",
                                flush=True,
                            )
                    return grad
            elif not cov_prev_valid:
                assert 0, (
                    f"SAME assert(0): cov_prev_valid_{adapter} is falsy (non-tensor) "
                    f"(layer={self.layer_id}, expert={expert_id})."
                )

            device = grad.device
            U_prev = getattr(self, f"cov_U_prev_{adapter}").to(device)
            S_prev = getattr(self, f"cov_S_prev_{adapter}").to(device)
            
            # 1) Effective rank k via energy threshold
            energy = S_prev ** 2
            total_energy = energy.sum()
            if total_energy < 1e-10:
                assert(0)
                
            cumsum = torch.cumsum(energy, dim=0)
            ratio = cumsum / (total_energy + 1e-10)
            thr = self.cumulative_energy_ratio
            k = (ratio <= thr).sum().item() + 1
            k = max(1, min(k, len(S_prev)))  # clamp k to [1, len(S_prev)]
            
            # 2) Curvature-aware scales
            mu = self.curvature_mu  
            inv_S = 1.0 / (S_prev[:k] + mu)      # in-subspace scaling
            perp_scale = 1.0 / mu                 # orthogonal scaling
            
            # 3) Project and scale gradient
            grad_t = grad.t()  # [d_in, d_out]
            U_k = U_prev[:, :k]  # [d_in, k]
            
            # Parallel part: V_k @ diag(inv_S) @ V_k.T @ grad.T
            proj_parallel = U_k @ (torch.diag(inv_S) @ (U_k.t() @ grad_t))
            
            # Perpendicular part: (1/μ) * (I - V_k @ V_k.T) @ grad.T
            proj_perp = grad_t - U_k @ (U_k.t() @ grad_t)
            proj_perp_scaled = proj_perp * perp_scale
            
            # 4) Combine
            scaled_grad_t = proj_parallel + proj_perp_scaled

            grad_norm_before = grad.norm().item()
            scaled_grad = scaled_grad_t.t()

            grad.data.copy_(scaled_grad.data)
            grad_norm_after = grad.norm().item()

            # if grad_norm_before > 1e-10:
            #     print(
            #         f"[HOOK] Layer{self.layer_id} {name} Expert{expert_id} | "
            #         f"k={k}/{len(S_prev)}, mu={mu:.1e} | "
            #         f"grad_norm: {grad_norm_before:.4f} → {grad_norm_after:.4f} "
            #         f"(ratio={grad_norm_after / (grad_norm_before + 1e-30):.2f}x)",
            #         flush=True,
            #     )
            return scaled_grad
        return hook
    

    
    def _apply_topk_sparsification(self, router_probs, router, k=2):
        probs = router_probs * (router.to(router_probs.device))
        #probs = router_probs
        probs = probs / (probs.sum() + 1e-8)
        if k < self.expert_num:
            topk_vals, topk_idx = torch.topk(probs, k)
            final_probs = torch.zeros_like(probs)
            final_probs[topk_idx] = probs[topk_idx]
            final_probs = final_probs / (final_probs.sum() + 1e-8)
        else:
            final_probs = probs
        return final_probs
    
    def forward(self, x: torch.Tensor, **kwargs):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.r[self.active_adapter] > 0:
            base_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            return _same_forward_with_routing(self, x, base_result)
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def _update_router_covariance(self, x_flat, adapter):
        """Online covariance / spectral stats for router hooks.

        Must not build autograd edges from ``x_flat``: with gradient checkpointing,
        the first forward can take the full QR branch while recomputation skips it
        (different ``current_step`` / ``cov_alpha``), which triggers
        ``CheckpointError`` (mismatching saved tensor counts).
        """
        device = x_flat.device
        batch_size = x_flat.size(0)

        cov_alpha = getattr(self, f"cov_alpha_{adapter}")

        with torch.no_grad():
            if cov_alpha == 0 or self.current_step % 20 == 0:
                d = x_flat.size(-1)
                k = min(self.max_components, d)

                x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)

                original_dtype = x_flat.dtype
                if original_dtype in [torch.bfloat16, torch.float16]:
                    compute_dtype = torch.float32
                else:
                    compute_dtype = original_dtype

                Q = torch.randn(d, k, device=device, dtype=compute_dtype)
                x_centered_comp = x_centered.to(compute_dtype)
                Y = x_centered_comp.t() @ (x_centered_comp @ Q)

                U_approx, _ = torch.linalg.qr(Y, mode="reduced")
                U_approx = U_approx.to(original_dtype)

                proj_x = x_centered @ U_approx
                S_approx = torch.sqrt(torch.sum(proj_x ** 2, dim=0) / (batch_size - 1 + 1e-8))
                S_approx = S_approx.to(original_dtype)

                setattr(self, f"cov_U_{adapter}", U_approx)
                setattr(self, f"cov_S_{adapter}", S_approx)

            setattr(self, f"cov_alpha_{adapter}", cov_alpha + batch_size)

    def _update_activation_metrics(self, routing_probs, x_flat):
        adapter = self.active_adapter
        device = x_flat.device
        util = getattr(self, f"utilization_{adapter}").to(device)
        udt = util.dtype
        batch_util = routing_probs.mean(dim=0).to(device=device, dtype=udt)
        new_util = 0.95 * util + 0.05 * batch_util
        setattr(self, f"utilization_{adapter}", new_util.detach())

        input_energy = torch.norm(x_flat, dim=1, keepdim=True) ** 2
        weighted_energy = routing_probs.to(dtype=udt) * input_energy.to(dtype=udt)
        batch_importance = weighted_energy.mean(dim=0)
        importance = getattr(self, f"importance_{adapter}").to(device=device, dtype=udt)
        new_importance = 0.95 * importance + 0.05 * batch_importance
        setattr(self, f"importance_{adapter}", new_importance.detach())

    def _apply_adaptive_freezing(self):
        adapter = self.active_adapter
        scores = self._compute_activation_scores(adapter)
        tau_score = self.tau_score
        current_task_expert = min(self.cur_task, self.expert_num - 1)
        masks = torch.ones(self.expert_num, device=scores.device, dtype=scores.dtype)
        low_score_mask = scores < tau_score
        #make sure a least one expert is activate
        low_score_mask[current_task_expert] = False
        masks[low_score_mask] = 0.0
        other_experts_mask = torch.ones(self.expert_num, dtype=torch.bool, device=scores.device)
        other_experts_mask[current_task_expert] = False
        all_other_frozen = low_score_mask[other_experts_mask].all()
        if all_other_frozen:
            self.all_frozen = True
        setattr(self, f"expert_masks_{adapter}", masks.detach())

    def _compute_activation_scores(self, adapter):
        util = getattr(self, f"utilization_{adapter}")
        imp = getattr(self, f"importance_{adapter}")
        
        device = util.device
        
        def normalize(t):
            t = t.to(device)
            if t.max() == t.min():
                return torch.zeros_like(t)
            return (t - t.min()) / (t.max() - t.min() + 1e-8)
        
        util_norm = normalize(util)
        imp_norm = normalize(imp)
        return util_norm - imp_norm

    def _get_masked_routing(self, routing_probs):
        adapter = self.active_adapter
        device = routing_probs.device
        rdtype = routing_probs.dtype

        masks = getattr(self, f"expert_masks_{adapter}").to(device=device, dtype=rdtype)

        masked = routing_probs * masks
        masked_sum = masked.sum(dim=-1, keepdim=True)
        if masked_sum.min() == 0:
            current_task_expert = min(self.cur_task, self.expert_num - 1)
            masked[:, current_task_expert] = torch.ones_like(masked[:, current_task_expert])
            masked_sum = masked.sum(dim=-1, keepdim=True)

        return masked / masked_sum

    def save_task_covariance_snapshot(self, adapter):
        U_curr = getattr(self, f"cov_U_{adapter}").clone()
        S_curr = getattr(self, f"cov_S_{adapter}").clone()
        k = (S_curr > 1e-10).sum().item()
        
        if k > 0:
            U_prev = torch.zeros_like(getattr(self, f"cov_U_prev_{adapter}"))
            S_prev = torch.zeros_like(getattr(self, f"cov_S_prev_{adapter}"))
            U_prev[:, :k] = U_curr[:, :k]
            S_prev[:k] = S_curr[:k]
            setattr(self, f"cov_U_prev_{adapter}", U_prev)
            setattr(self, f"cov_S_prev_{adapter}", S_prev)
            setattr(self, f"cov_prev_valid_{adapter}", torch.tensor(True))
        else:
            setattr(self, f"cov_prev_valid_{adapter}", torch.tensor(False))
            print(f"[Layer {self.layer_id}] No valid covariance to save")

    def reset_for_new_task(self, adapter):
        setattr(self, f"cov_alpha_{adapter}", torch.tensor(0.0))
        setattr(self, f"utilization_{adapter}", torch.ones(self.expert_num) / self.expert_num)
        setattr(self, f"expert_masks_{adapter}", torch.ones(self.expert_num))
        
        if self.training:
            lora_a_module = self.lora_A[adapter]
            lora_b_module = self.lora_B[adapter]
            for expert_id in range(self.expert_num):
                for param in lora_a_module.loraA[expert_id].parameters():
                    param.requires_grad_(True)
                for param in lora_b_module.loraB[expert_id].parameters():
                    param.requires_grad_(True)


_SAME_LINEAR_SHARED_METHODS = (
    "merge",
    "unmerge",
    "_register_same_hooks",
    "_spectral_aware_router_hook",
    "_make_curvature_hook",
    "_apply_topk_sparsification",
    "_update_router_covariance",
    "_update_activation_metrics",
    "_apply_adaptive_freezing",
    "_compute_activation_scores",
    "_get_masked_routing",
    "save_task_covariance_snapshot",
    "reset_for_new_task",
)


if is_bnb_available():

    class SAMELinear8bitLt(bnb.nn.Linear8bitLt, SAMELayer):
        """8-bit quantized SAME linear (bitsandbytes) with covariance carry-over buffers."""

        _hook_skip_cov_prev_warned: bool = False

        def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            train_signal: bool = False,
            layer_id: int = 0,
            **kwargs,
        ):
            cfg = _same_pop_layer_kwargs(kwargs)
            init_lora_weights = cfg["init_lora_weights"]
            self.expert_num = cfg["expert_num"]
            self.te_dim = cfg["te_dim"]
            self.cur_task = cfg["cur_task"]
            self.layer_id = layer_id
            self.max_components = cfg["max_components"]
            self.window_size = cfg["window_size"]
            self.curvature_mu = cfg["curvature_mu"]
            self.tau_score = cfg["tau_score"]
            self.cumulative_energy_ratio = cfg["cumulative_energy_ratio"]
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            SAMELayer.__init__(
                self,
                in_features=in_features,
                out_features=out_features,
                expert_num=self.expert_num,
                cur_task=self.cur_task,
                training=train_signal,
                layer_id=layer_id,
            )
            self.training_signal = train_signal
            self.fan_in_fan_out = False
            _same_init_runtime_state(
                self, adapter_name, in_features, r, lora_alpha, lora_dropout, init_lora_weights, train_signal
            )

        def forward(self, x: torch.Tensor, **kwargs):
            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return bnb.nn.Linear8bitLt.forward(self, x)
            if self.r[self.active_adapter] <= 0:
                return bnb.nn.Linear8bitLt.forward(self, x)
            base_result = bnb.nn.Linear8bitLt.forward(self, x)
            return _same_forward_with_routing(self, x, base_result)

    for _name in _SAME_LINEAR_SHARED_METHODS:
        setattr(SAMELinear8bitLt, _name, getattr(SAMELinear, _name))


class SAMELinearA(nn.Module):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([SAMEExpert(in_features, out_features // expert_num) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x, routing_weights):
        rw = routing_weights.to(dtype=x.dtype)
        output = self.loraA[0](x) * rw[0]
        for i in range(1, self.expert_num):
            output += self.loraA[i](x) * rw[i]
        return output

class SAMELinearB(nn.Module):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([SAMEExpert(in_features // expert_num, out_features) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x, routing_weights):
        rw = routing_weights.to(dtype=x.dtype)
        output = self.loraB[0](x) * rw[0]
        for i in range(1, self.expert_num):
            output += self.loraB[i](x) * rw[i]
        return output

class SAMEExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)