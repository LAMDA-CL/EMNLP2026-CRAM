from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.backbone.registry import get_routing_feature_dim
from method.base.routing_utils import extract_routing_image_features
from method.base.context import CLContext
from method.base.integration import CLIntegration

_PRIOR_VEC_KEY = "_prior_expert_vec"
_LEGACY_PRIOR_KEY = "_last_routing"

# Substrings matched when restoring carry-over buffers from checkpoint state.
_ROUTER_CARRY_BUFFER_MARKERS: Tuple[str, ...] = (
    "cov_U_prev",
    "cov_S_prev",
    "importance",
    "cov_prev_valid",
)


def merge_tensor_bundles(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
    *,
    on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "raise",
) -> Dict[str, torch.Tensor]:
    dup = set(first) & set(second)
    if dup and on_duplicate_key == "raise":
        sample = sorted(dup)[:32]
        raise ValueError(
            f"merge_tensor_bundles: duplicate keys ({len(dup)} total), e.g. {sample!r}"
        )
    if on_duplicate_key == "prefer_first":
        out = dict(second)
        out.update(first)
        return out
    out = dict(first)
    out.update(second)
    return out


def read_merge_write_safetensors(
    safetensors_path: str,
    extra: Dict[str, torch.Tensor],
    *,
    on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "prefer_second",
) -> bool:
    if not extra:
        return False

    from safetensors.torch import load_file, save_file

    if not os.path.isfile(safetensors_path):
        # PEFT may only have adapter_model.bin on some runs; still persist SAME extras.
        save_file(extra, safetensors_path)
        return True

    base: Dict[str, torch.Tensor] = load_file(safetensors_path)
    merged = merge_tensor_bundles(base, extra, on_duplicate_key=on_duplicate_key)
    save_file(merged, safetensors_path)
    return True

class PromptIntegration(CLIntegration):

    def __init__(self, config: Any):
        super().__init__(config)
        self.num_prompt_tokens: int = int(getattr(config, "num_prompt_tokens", 8))
        self.virtual_tokens: Optional[int] = getattr(config, "virtual_tokens", None)
        if self.virtual_tokens is not None:
            self.virtual_tokens = int(self.virtual_tokens)

    def initialize_model(self, model: nn.Module) -> None:
        return

    def on_input_prep(
        self,
        model: nn.Module,
        args: tuple,
        kwargs: dict,
        context: CLContext,
    ) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "num_prompt_tokens": self.num_prompt_tokens,
        }
        if self.virtual_tokens is not None:
            out["virtual_tokens"] = self.virtual_tokens
        return out

class RouterIntegration(CLIntegration):
    _PRISM_SAME_PREFIX = "prism.same."
    _LEGACY_SAME_PREFIXES = ("mcitbox.same.",)

    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim: int = int(get_routing_feature_dim(getattr(config, "backbone", None)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))

        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None

        self._model_ref: Any = None
        self._prior_expert_vec: Optional[torch.Tensor] = None
        self.mixture_logit_scale: float = float(getattr(config, "mixture_logit_scale", getattr(config, "routing_softmax_scale", 28.0)))
        self.peft_expert_layer_name: str = str(
            getattr(
                config,
                "peft_expert_layer_name",
                getattr(config, "peft_routing_module_name", "SAMELinear"),
            )
        )

    def initialize_model(self, model: Any) -> None:
        self._model_ref = model
        self._ensure_prototypes_on_model(model)

    def merge_extra_into_adapter_safetensors(
        self,
        output_dir: str,
        extra: Dict[str, torch.Tensor],
        *,
        on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "prefer_second",
    ) -> bool:
        st_path = os.path.join(output_dir, "adapter_model.safetensors")
        return read_merge_write_safetensors(
            st_path, extra, on_duplicate_key=on_duplicate_key
        )

    @staticmethod
    def _state_has_nonempty_anchor_lists(state: Dict[str, Any]) -> bool:
        ia, ta = state.get("image_anchors"), state.get("text_anchors")
        return (
            isinstance(ia, (list, tuple))
            and isinstance(ta, (list, tuple))
            and len(ia) > 0
            and len(ta) > 0
        )

    @staticmethod
    def _normalize_router_state_keys(state: Dict[str, Any]) -> Dict[str, Any]:
        if not any(isinstance(k, str) and k.startswith("_base_model.") for k in state):
            return state
        out: Dict[str, Any] = {}
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("_base_model."):
                out[k[len("_base_model.") :]] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _canonical_router_buffer_key(name: str) -> str:
        """Align buffer names across CLModel / PeftModel / saved checkpoint key variants."""
        s = name
        if s.startswith("_base_model."):
            s = s[len("_base_model.") :]
        while "base_model.model.model." in s:
            s = s.replace("base_model.model.model.", "base_model.model.", 1)
        return s

    @classmethod
    def _build_router_tensor_index(
        cls,
        state: Dict[str, Any],
        buffer_markers: Tuple[str, ...] = _ROUTER_CARRY_BUFFER_MARKERS,
    ) -> Dict[str, torch.Tensor]:
        idx: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if not isinstance(k, str) or not isinstance(v, torch.Tensor):
                continue
            if not any(m in k for m in buffer_markers):
                continue
            idx[cls._canonical_router_buffer_key(k)] = v
        return idx

    def _read_router_checkpoint_files(
        self,
        load_dir: str,
        *,
        carryover_bin: str = "same_state.bin",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str, str]:
        st_path = os.path.join(load_dir, "adapter_model.safetensors")
        bin_path = os.path.join(load_dir, carryover_bin)

        state_from_bin: Optional[Dict[str, Any]] = None
        if os.path.isfile(bin_path):
            blob = torch.load(bin_path, map_location="cpu")
            if isinstance(blob, dict):
                state_from_bin = self._normalize_router_state_keys(blob)

        state_from_st: Optional[Dict[str, Any]] = None
        if os.path.isfile(st_path):
            from safetensors.torch import load_file

            flat = load_file(st_path)
            state_from_st = self._tensor_bundle_to_same_state(flat)

        return state_from_bin, state_from_st, bin_path, st_path

    def _merge_router_checkpoint_states(
        self,
        state_from_bin: Optional[Dict[str, Any]],
        state_from_st: Optional[Dict[str, Any]],
        *,
        bin_path: str = "",
        st_path: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Pick primary checkpoint source; prefer non-empty anchor lists, then merge the other file."""
        state: Optional[Dict[str, Any]] = None
        p = ""

        if state_from_bin and self._state_has_nonempty_anchor_lists(state_from_bin):
            state = dict(state_from_bin)
            p = bin_path
            if state_from_st:
                for k, v in state_from_st.items():
                    if k not in state:
                        state[k] = v
        elif state_from_st and self._state_has_nonempty_anchor_lists(state_from_st):
            state = dict(state_from_st)
            p = st_path
            if state_from_bin:
                for k, v in state_from_bin.items():
                    if k not in state:
                        state[k] = v
        elif state_from_bin:
            state = dict(state_from_bin)
            p = bin_path
        elif state_from_st:
            state = dict(state_from_st)
            p = st_path

        return state, p

    def carry_buffer_markers(self) -> Tuple[str, ...]:
        """Substrings for carry-over buffers when loading ``same_state.bin`` / safetensors extras."""
        return _ROUTER_CARRY_BUFFER_MARKERS

    def _similarities_to_mixture(self, sims: torch.Tensor) -> torch.Tensor:
        x = sims.to(dtype=torch.float32).reshape(-1)
        if x.numel() != self.task_num:
            raise ValueError(
                f"_similarities_to_mixture: expected {self.task_num} scores, got {x.numel()}"
            )
        logits = x * self.mixture_logit_scale
        return F.softmax(logits, dim=0)

    def _ensure_prototypes_on_model(self, model: Any) -> None:
        device = next(model.parameters()).device
        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)

        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.task_num = self.task_num

    def _extract_clip_features(
        self, model: Any, images: Any, input_ids: Any, clip_tokenizer: Any, text_tower: Any
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        device = images.device if images is not None else next(model.parameters()).device

        image_feat: Optional[torch.Tensor] = None
        if images is not None:
            try:
                image_feat = extract_routing_image_features(model, images)
            except RuntimeError:
                image_feat = None

        if input_ids is None:
            text_feat = torch.randn(1, self.feature_dim, device=device)
        else:
            main_tokenizer = getattr(model, "tokenizer", None)
            if main_tokenizer is None:
                _base_model = getattr(model, "_base_model", None)
                if _base_model is not None:
                    main_tokenizer = getattr(_base_model, "tokenizer", None)
            tok = main_tokenizer or clip_tokenizer
            input_pad = np.where(
                input_ids.cpu().detach().numpy() != -200,
                input_ids.cpu().detach().numpy(),
                tok.pad_token_id,
            )
            decoded = tok.batch_decode(input_pad, skip_special_tokens=True)
            decoded_hidden = ["\n".join(d.split("\n")[1:]) for d in decoded]
            decoded_clip = [d.split(" ASSISTANT")[0] for d in decoded_hidden]
            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_feat = text_tower(clip_inputs)
                text_feat = text_feat[0] if isinstance(text_feat, tuple) else text_feat
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        return image_feat, text_feat

    def _clip_tokenizer_and_text_tower(self, model: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Resolve HiDe-style CLIP tokenizer + text tower across CLModel / PEFT / Llava nesting."""

        def _unwrap_tower(obj: Any) -> Any:
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                return obj[0]
            return obj

        def _tower_from(m: Any) -> Any:
            if m is None:
                return None
            tt = _unwrap_tower(getattr(m, "text_tower", None))
            if tt is not None:
                return tt
            gt = getattr(m, "get_text_tower", None)
            if callable(gt):
                return _unwrap_tower(gt())
            return None

        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = _tower_from(model)
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            clip_tokenizer = clip_tokenizer or getattr(_base_model, "clip_tokenizer", None)
            text_tower = text_tower or _tower_from(_base_model)
            if hasattr(_base_model, "base_model"):
                clip_tokenizer = clip_tokenizer or getattr(_base_model.base_model, "clip_tokenizer", None)
                text_tower = text_tower or _tower_from(_base_model.base_model)
            if hasattr(_base_model, "model") and text_tower is None:
                text_tower = _tower_from(getattr(_base_model, "model"))
            inner = getattr(_base_model, "model", None)
            if inner is not None and hasattr(inner, "model") and text_tower is None:
                text_tower = _tower_from(getattr(inner, "model"))
        return clip_tokenizer, text_tower

    def _compute_cosine_similarities(
        self, image_feat: Optional[torch.Tensor], text_feat: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        text_sims: list = []
        for t in range(self.task_num):
            ta = self.text_anchors[t].to(device)
            text_sims.append(F.cosine_similarity(text_feat.unsqueeze(1), ta.unsqueeze(0), dim=2).max())
        text_sims_t = torch.stack(text_sims).to(device=device, dtype=torch.float32)

        if image_feat is not None:
            image_sims: list = []
            for t in range(self.task_num):
                ia = self.image_anchors[t].to(device)
                image_sims.append(F.cosine_similarity(image_feat.unsqueeze(1), ia.unsqueeze(0), dim=2).max())
            image_sims_t = torch.stack(image_sims).to(device=device, dtype=torch.float32)
            return 0.2 * image_sims_t + 0.8 * text_sims_t
        return text_sims_t

    def _update_running_prototypes(
        self,
        image_feat: Optional[torch.Tensor],
        text_feat: torch.Tensor,
        task_id: int,
    ) -> None:
        bs = int(text_feat.shape[0])
        with torch.no_grad():
            if image_feat is not None:
                old = self.image_anchors[task_id].data.clone()
                cnt = self.image_boundary[task_id].data.clone()
                new_cnt = cnt + bs
                self.image_anchors[task_id].data.copy_((old * cnt + image_feat.sum(dim=0)) / new_cnt)
                self.image_boundary[task_id].data.copy_(new_cnt)
            oldt = self.text_anchors[task_id].data.clone()
            cntt = self.text_boundary[task_id].data.clone()
            new_cntt = cntt + bs
            self.text_anchors[task_id].data.copy_((oldt * cntt + text_feat.sum(dim=0)) / new_cntt)
            self.text_boundary[task_id].data.copy_(new_cntt)

    def _batch_prepare(
        self,
        model: Any,
        images: Any,
        input_ids: Any,
        context: CLContext,
    ) -> None:
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return

        if input_ids is None or (hasattr(input_ids, "shape") and input_ids.shape[1] <= 1):
            if self._prior_expert_vec is not None:
                self._write_expert_mix_to_modules(model, self._prior_expert_vec)
            return

        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
        device = text_feat.device

        if (
            model.training
            and torch.is_grad_enabled()
            and context.task_id is not None
            and 0 <= int(context.task_id) < self.task_num
        ):
            tid = int(context.task_id)
            self._update_running_prototypes(image_feat, text_feat, tid)
            mix = torch.zeros(self.task_num, device=device, dtype=torch.float32)
            mix[tid] = 1.0
            self._write_expert_mix_to_modules(model, mix)
            self._prior_expert_vec = mix.detach()
            return

        sims_t = self._compute_cosine_similarities(image_feat, text_feat, device)
        mix = self._similarities_to_mixture(sims_t)
        self._write_expert_mix_to_modules(model, mix)
        self._prior_expert_vec = mix.detach()

    def on_input_prep(
        self, model: Any, args: tuple, kwargs: dict, context: CLContext
    ) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None
        self._batch_prepare(model, images, input_ids, context)

    def pre_generate_hook(
        self, model: Any, input_ids: Any, images: Any, context: CLContext
    ) -> bool:
        self._batch_prepare(model, images, input_ids, context)
        return True

    def _matches_peft_expert_layer(self, module: Any) -> bool:
        name = module.__class__.__name__
        target = self.peft_expert_layer_name
        return name == target or name.startswith(f"{target}")

    def _write_expert_mix_to_modules(self, model: Any, mix: torch.Tensor) -> None:
        mix = mix.detach()
        for module in model.modules():
            if not self._matches_peft_expert_layer(module):
                continue
            target_len = int(getattr(module, "expert_num", mix.numel()))
            vec = mix
            if mix.numel() != target_len:
                vec = torch.zeros(target_len, dtype=mix.dtype, device=mix.device)
                copy_len = min(target_len, mix.numel())
                vec[:copy_len] = mix[:copy_len]
                vec = vec / (vec.sum() + 1e-8)
            module.router = vec.to(device=next(module.parameters()).device, dtype=torch.float32)

    def sync_anchors_to_model(self, model: Any) -> None:
        if model is None:
            return
        if self.image_anchors is not None:
            object.__setattr__(model, "image_anchors", self.image_anchors)
        if self.text_anchors is not None:
            object.__setattr__(model, "text_anchors", self.text_anchors)
        if self.image_boundary is not None:
            object.__setattr__(model, "image_boundary", self.image_boundary)
        if self.text_boundary is not None:
            object.__setattr__(model, "text_boundary", self.text_boundary)

    def load_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.image_anchors is not None:
            out["image_anchors"] = [p.detach().cpu().clone() for p in self.image_anchors]
        if self.text_anchors is not None:
            out["text_anchors"] = [p.detach().cpu().clone() for p in self.text_anchors]
        if self.image_boundary is not None:
            out["image_boundary"] = [p.detach().cpu().clone() for p in self.image_boundary]
        if self.text_boundary is not None:
            out["text_boundary"] = [p.detach().cpu().clone() for p in self.text_boundary]
        if self._prior_expert_vec is not None:
            t = self._prior_expert_vec.detach().cpu().contiguous().clone()
            out[_PRIOR_VEC_KEY] = t
            out[_LEGACY_PRIOR_KEY] = t.clone()
        return out

    def restore_state(self, state: Dict[str, Any], model: Optional[Any] = None) -> Tuple[bool, bool]:
        """Restore SAME/Router prototypes and auxiliary tensors.

        Returns:
            ``(anchor_lists_ok, aux_ok)`` — ``anchor_lists_ok`` is True only when **both**
            ``image_anchors`` and ``text_anchors`` non-empty lists were copied.
            ``aux_ok`` is True when boundaries and/or the routing prior were restored.
        """
        anchor_lists_ok = False
        aux_ok = False

        if (
            self.image_anchors is not None
            and self.text_anchors is not None
            and self._state_has_nonempty_anchor_lists(state)
        ):
            ia = state["image_anchors"]
            ta = state["text_anchors"]
            for i, p in enumerate(ia):
                if i < len(self.image_anchors) and isinstance(p, torch.Tensor):
                    self.image_anchors[i].data.copy_(
                        p.to(device=self.image_anchors[i].device, dtype=self.image_anchors[i].dtype)
                    )
            for i, p in enumerate(ta):
                if i < len(self.text_anchors) and isinstance(p, torch.Tensor):
                    self.text_anchors[i].data.copy_(
                        p.to(device=self.text_anchors[i].device, dtype=self.text_anchors[i].dtype)
                    )
            anchor_lists_ok = True

        if "image_boundary" in state and isinstance(state["image_boundary"], (list, tuple)) and self.image_boundary is not None:
            for i, p in enumerate(state["image_boundary"]):
                if i < len(self.image_boundary) and isinstance(p, torch.Tensor):
                    self.image_boundary[i].data.copy_(
                        p.to(device=self.image_boundary[i].device, dtype=self.image_boundary[i].dtype)
                    )
            aux_ok = True
        if "text_boundary" in state and isinstance(state["text_boundary"], (list, tuple)) and self.text_boundary is not None:
            for i, p in enumerate(state["text_boundary"]):
                if i < len(self.text_boundary) and isinstance(p, torch.Tensor):
                    self.text_boundary[i].data.copy_(
                        p.to(device=self.text_boundary[i].device, dtype=self.text_boundary[i].dtype)
                    )
            aux_ok = True

        pv = None
        if _PRIOR_VEC_KEY in state and isinstance(state[_PRIOR_VEC_KEY], torch.Tensor):
            pv = state[_PRIOR_VEC_KEY]
        elif _LEGACY_PRIOR_KEY in state and isinstance(state[_LEGACY_PRIOR_KEY], torch.Tensor):
            pv = state[_LEGACY_PRIOR_KEY]
        if pv is not None:
            self._prior_expert_vec = pv.clone()
            aux_ok = True

        if model is not None:
            self._model_ref = model
            self.sync_anchors_to_model(model)
        return anchor_lists_ok, aux_ok

    def restore_state_any(self, state: Dict[str, Any], model: Optional[Any] = None) -> bool:
        """True if prototypes, boundaries, or prior were restored."""
        a_ok, x_ok = self.restore_state(state, model=model)
        return bool(a_ok or x_ok)

    def _same_state_to_tensor_bundle(self, same_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pref = self._PRISM_SAME_PREFIX
        out: Dict[str, torch.Tensor] = {}
        list_keys = ("image_anchors", "text_anchors", "image_boundary", "text_boundary")
        for lk in list_keys:
            if lk not in same_state:
                continue
            seq = same_state[lk]
            if not isinstance(seq, (list, tuple)):
                continue
            for i, t in enumerate(seq):
                if isinstance(t, torch.Tensor):
                    out[f"{pref}{lk}.{i}"] = t.detach().cpu().contiguous().clone()
        for pk in ("_prior_expert_vec", "_last_routing"):
            if pk in same_state and isinstance(same_state[pk], torch.Tensor):
                out[f"{pref}{pk}"] = same_state[pk].detach().cpu().contiguous().clone()
        for k, v in same_state.items():
            if k in list_keys or k in ("_prior_expert_vec", "_last_routing"):
                continue
            if isinstance(v, torch.Tensor):
                safe = k.replace(".", "__DOT__")
                out[f"{pref}buf.{safe}"] = v.detach().cpu().contiguous().clone()
        return out

    def _tensor_bundle_to_same_state(self, flat: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        for pref in (self._PRISM_SAME_PREFIX,) + self._LEGACY_SAME_PREFIXES:
            sub = {k: v for k, v in flat.items() if k.startswith(pref)}
            if not sub:
                continue
            state: Dict[str, Any] = {}
            list_keys = ("image_anchors", "text_anchors", "image_boundary", "text_boundary")
            for lk in list_keys:
                pref_list = f"{pref}{lk}."
                idx_tensors: Dict[int, torch.Tensor] = {}
                for k, v in sub.items():
                    if not k.startswith(pref_list):
                        continue
                    tail = k[len(pref_list) :]
                    if tail.isdigit():
                        idx_tensors[int(tail)] = v
                if idx_tensors:
                    mx = max(idx_tensors)
                    lst = [idx_tensors[i] for i in range(mx + 1) if i in idx_tensors]
                    if lst:
                        state[lk] = lst
            for pk in ("_prior_expert_vec", "_last_routing"):
                key = f"{pref}{pk}"
                if key in sub:
                    state[pk] = sub[key]
            buf_prefix = f"{pref}buf."
            for k, v in sub.items():
                if not k.startswith(buf_prefix):
                    continue
                orig = k[len(buf_prefix) :].replace("__DOT__", ".")
                state[orig] = v
            return state
        return {}

    def print_carryover_restore_summary(
        self, path: str, state: Dict[str, Any], tag: str = "[Router]"
    ) -> None:
        pass

    def save_carryover_file(self, output_dir: str, filename: str = "carryover_state.bin") -> bool:
        os.makedirs(output_dir, exist_ok=True)
        state = self.load_state()
        if not state:
            return False
        torch.save(state, os.path.join(output_dir, filename))
        return True

    def load_carryover_file(
        self, load_dir: str, model: Optional[Any] = None, filename: str = "carryover_state.bin"
    ) -> bool:
        p = os.path.join(load_dir, filename)
        if not os.path.exists(p):
            return False
        state = torch.load(p, map_location="cpu")
        if not isinstance(state, dict):
            return False
        ok = self.restore_state_any(state, model=model)
        if ok:
            self.print_carryover_restore_summary(p, state)
        return ok


