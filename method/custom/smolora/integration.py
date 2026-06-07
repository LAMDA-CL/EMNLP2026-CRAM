from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config.backbone.registry import get_routing_feature_dim
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
    from PEFT.tuners.custom.smolora import SMoLoraConfig, SMoLoraModel

    register_peft_extension(
        peft_type="SMOLORA",
        config_cls=SMoLoraConfig,
        tuner_model_cls=SMoLoraModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("smolora")
class SmoloraIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.smolora_expert_num: int = int(getattr(config, "smolora_expert_num", 8))
        self._ins_emb_path: Optional[str] = getattr(config, "ins_emb_path", None) or None
        if self._ins_emb_path and not os.path.isabs(self._ins_emb_path):
            root = getattr(config, "project_root", None) or getattr(config, "repo_root", None)
            if root:
                self._ins_emb_path = os.path.join(str(root), self._ins_emb_path)
        self._author_ins_emb_list: Optional[List[List[float]]] = None
        self._use_builtin_sentence_ins_emb: bool = False
        self._st_model_name: str = str(
            getattr(config, "smolora_sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        if self._ins_emb_path:
            t = self._load_ins_emb_tensor(self._ins_emb_path)
            self._author_ins_emb_list = t.detach().cpu().float().tolist()
            inferred_d = int(t.shape[-1])
            self.ins_emb_dim: int = int(getattr(config, "ins_emb_dim", inferred_d))
            if self.ins_emb_dim != inferred_d:
                raise ValueError(
                    f"ins_emb_dim={self.ins_emb_dim} does not match last dim {inferred_d} from {self._ins_emb_path}."
                )
        else:
            self._use_builtin_sentence_ins_emb = bool(
                getattr(config, "smolora_builtin_sentence_ins_emb", True)
            )
            if self._use_builtin_sentence_ins_emb:
                self.ins_emb_dim: int = int(getattr(config, "ins_emb_dim", 384))
            else:
                self.ins_emb_dim: int = int(getattr(config, "ins_emb_dim", get_routing_feature_dim(getattr(config, "backbone", None))))
        self._default_ins_type: int = int(getattr(config, "ins_type", 0))
        self._clip_no_grad: bool = bool(getattr(config, "smolora_clip_no_grad", True))
        self._model_ref = None

    @staticmethod
    def _load_ins_emb_tensor(path: str) -> torch.Tensor:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, torch.Tensor):
            t = obj.float()
        elif isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj).float()
        else:
            t = torch.as_tensor(obj).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2:
            raise ValueError(f"Instruction embedding tensor must be 2D [T, D]; got shape {tuple(t.shape)} for {path}")
        return t

    def initialize_model(self, model) -> None:
        self._model_ref = model
        for _, p in model.named_parameters():
            p.requires_grad = False

        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.smolora import SMoLoraConfig
        from PEFT.utils import TaskType

        target_modules = self._find_target_modules(model)
        r = int(getattr(self.config, "lora_r", 64))
        if r % self.smolora_expert_num != 0:
            raise ValueError(
                f"lora_r={r} must be divisible by smolora_expert_num={self.smolora_expert_num}."
            )

        if self._use_builtin_sentence_ins_emb and self._author_ins_emb_list is None:
            from method.custom.smolora.builtin_ins_emb import compute_default_ins_emb_matrix

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = compute_default_ins_emb_matrix(model_name=self._st_model_name, device=dev)
            self._author_ins_emb_list = t.detach().cpu().float().tolist()
            self.ins_emb_dim = int(t.shape[-1])

        peft_config = SMoLoraConfig(
            target_modules=target_modules,
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            bias=str(getattr(self.config, "lora_bias", "none")),
            task_type=TaskType.CAUSAL_LM_SMOLORA,
            expert_num=self.smolora_expert_num,
            ins_type=int(getattr(self.config, "cur_task", self._default_ins_type)),
            ins_emb_dim=self.ins_emb_dim,
            ins_emb=self._author_ins_emb_list,
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, peft_config)

        object.__setattr__(model, "_smolora_last_instruction_feat", None)
        tid = int(getattr(self.config, "cur_task", self._default_ins_type))
        self._propagate_ins_type(model, tid)

    def _find_target_modules(self, model) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def _propagate_ins_type(self, model: Any, ins_type: int) -> None:
        from PEFT.tuners.custom.smolora import SMoLoraLinear

        it = int(ins_type)
        for module in model.modules():
            if isinstance(module, SMoLoraLinear):
                module.ins_type = it

    def _clip_tokenizer_and_text_tower(self, model: Any) -> Tuple[Optional[Any], Optional[Any]]:
        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = getattr(model, "text_tower", None)
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            clip_tokenizer = clip_tokenizer or getattr(_base_model, "clip_tokenizer", None)
            text_tower = text_tower or getattr(_base_model, "text_tower", None)
            if hasattr(_base_model, "base_model"):
                clip_tokenizer = clip_tokenizer or getattr(_base_model.base_model, "clip_tokenizer", None)
                text_tower = text_tower or getattr(_base_model.base_model, "text_tower", None)
        return clip_tokenizer, text_tower

    def _decode_prompts_for_clip(self, model: Any, input_ids: torch.Tensor, clip_tokenizer: Any) -> List[str]:
        main_tokenizer = getattr(model, "tokenizer", None)
        if main_tokenizer is None:
            _base_model = getattr(model, "_base_model", None)
            if _base_model is not None:
                main_tokenizer = getattr(_base_model, "tokenizer", None)

        pad_id = main_tokenizer.pad_token_id if main_tokenizer is not None else clip_tokenizer.pad_token_id
        input_pad = np.where(
            input_ids.detach().cpu().numpy() != -200,
            input_ids.detach().cpu().numpy(),
            pad_id,
        )
        if main_tokenizer is not None:
            decoded_inputs = main_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        else:
            decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)

        decoded_hidden = ["\n".join(d.split("\n")[1:]) for d in decoded_inputs]
        decoded_clip = [d.split(" ASSISTANT")[0] for d in decoded_hidden]
        return decoded_clip

    def _encode_clip_text(
        self,
        text_tower: Any,
        clip_tokenizer: Any,
        decoded_clip: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        clip_inputs = clip_tokenizer(
            decoded_clip,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        if self._clip_no_grad:
            with torch.no_grad():
                text_feat = text_tower(clip_inputs)
        else:
            text_feat = text_tower(clip_inputs)
        if isinstance(text_feat, tuple):
            text_feat = text_feat[0]
        text_feat = text_feat.to(device=device, dtype=dtype)
        if text_feat.dim() == 3:
            text_feat = text_feat.mean(dim=1)
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        return text_feat

    def _compute_runtime_instruction_feat(
        self,
        model: Any,
        input_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return None

        if input_ids is None or (hasattr(input_ids, "shape") and input_ids.shape[1] <= 1):
            return getattr(model, "_smolora_last_instruction_feat", None)

        device = input_ids.device
        dtype = next(model.parameters()).dtype
        decoded_clip = self._decode_prompts_for_clip(model, input_ids, clip_tokenizer)
        feat = self._encode_clip_text(text_tower, clip_tokenizer, decoded_clip, device, dtype)
        if self._clip_no_grad:
            feat = feat.detach()
        object.__setattr__(model, "_smolora_last_instruction_feat", feat)
        return feat

    def _propagate_runtime_instruction_feat(self, model: Any, feat: Optional[torch.Tensor]) -> None:
        from PEFT.tuners.custom.smolora import SMoLoraLinear

        for module in model.modules():
            if isinstance(module, SMoLoraLinear):
                module._runtime_instruction_feat = feat

    def _resolve_task_id(self, model: Any, context: CLContext) -> int:
        tid = getattr(model, "cur_task", None)
        if tid is not None:
            return int(tid)
        if context.task_id is not None:
            return int(context.task_id)
        return int(getattr(self.config, "cur_task", self._default_ins_type))

    def _sync_smolora_inputs(self, model: Any, context: CLContext, input_ids: Any) -> None:
        self._propagate_ins_type(model, self._resolve_task_id(model, context))
        if self._author_ins_emb_list is not None:
            self._propagate_runtime_instruction_feat(model, None)
            return
        feat = self._compute_runtime_instruction_feat(model, input_ids)
        self._propagate_runtime_instruction_feat(model, feat)

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        input_ids = args[0] if args else None
        self._sync_smolora_inputs(model, context, input_ids)

    def pre_generate_hook(self, model: Any, input_ids: Any, images: Any, context: CLContext) -> bool:
        self._sync_smolora_inputs(model, context, input_ids)
        return True

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        return

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict:
        out = {
            "smolora_expert_num": self.smolora_expert_num,
            "ins_emb_dim": self.ins_emb_dim,
            "smolora_clip_no_grad": self._clip_no_grad,
        }
        if self._ins_emb_path:
            out["ins_emb_path"] = self._ins_emb_path
        out["smolora_builtin_sentence_ins_emb"] = self._use_builtin_sentence_ins_emb
        if self._use_builtin_sentence_ins_emb:
            out["smolora_sentence_transformer_model"] = self._st_model_name
        return out
