"""
modal_prompt: per-task soft prompts + prompt transform MLPs with
dual-modal (CLIP image + text) guided top-K prompt selection.

Training: only current task's prompt parameters and transform are trainable.
Inference: selects top-K prompts via modal guidance and prepends them.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.routing_utils import extract_routing_image_features
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft_model import PeftModelForCausalLMModalPrompt
    from PEFT.tuners.custom.modal_prompt import ModalPromptConfig, ModalPromptModel

    register_peft_extension(
        peft_type="MODAL_PROMPT",
        config_cls=ModalPromptConfig,
        tuner_model_cls=ModalPromptModel,
        task_type="CAUSAL_LM_MODAL_PROMPT",
        task_peft_model_cls=PeftModelForCausalLMModalPrompt,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("modal_prompt")
class Modal_promptIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim: int = int(getattr(config, "clip_feature_dim", 768))
        self.prefix_len: int = int(getattr(config, "prefix_len", 10))
        self.transfer_num: int = int(getattr(config, "transfer_num", 3))
        self.lam: float = float(getattr(config, "lam", 0.5))
        self.cur_task: int = int(getattr(config, "cur_task", 0))
        self._model_ref: Optional[Any] = None

        # Cosine similarity losses (set during on_input_prep, consumed in on_forward_end)
        self._image_cos_sim_loss: Optional[torch.Tensor] = None
        self._text_cos_sim_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_modal_prompt_model(self, model) -> Optional[Any]:
        from PEFT.tuners.custom.modal_prompt import ModalPromptModel

        root = getattr(model, "_base_model", None) or model
        for m in root.modules():
            if isinstance(m, ModalPromptModel):
                return m
        return None

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

    def _extract_clip_features(
        self, model, images, input_ids, clip_tokenizer, text_tower
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Extract CLIP image and text features. Returns (image_feat, text_feat)."""
        device = images.device if images is not None else next(model.parameters()).device

        # Image features
        if images is not None:
            try:
                image_feat = extract_routing_image_features(model, images)
            except RuntimeError:
                image_feat = None
        else:
            image_feat = None

        # Text features
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

    def _compute_prototypes(self, mpm: Any) -> torch.Tensor:
        """Compute prototype embeddings for COMPLETED tasks [0, cur_task).
        Each prototype = mean(prompt_transform_i(prompt_tokens_i)).
        Returns: (cur_task, feature_dim) — with cur_task = number of completed tasks.

        All returned prototypes are detached (past tasks are frozen).
        """
        prototypes = []
        for i in range(self.cur_task):
            raw_prompts = mpm.task_prompts[i].detach()
            with torch.no_grad():
                transformed = mpm.prompt_transforms[i](raw_prompts)
            proto = transformed.mean(dim=0)
            prototypes.append(proto)
        if prototypes:
            return torch.stack(prototypes, dim=0)  # (cur_task, feature_dim)
        if self.cur_task > 0:
            return torch.empty(0, self.feature_dim, device=mpm.task_prompts[0].device)
        return torch.empty(0, self.feature_dim)

    def _compute_current_prototype(self, mpm: Any) -> torch.Tensor:
        """Compute prototype for the CURRENT task (with gradients)."""
        tid = int(self.cur_task)
        raw_prompts = mpm.task_prompts[tid]  # retains gradients
        transformed = mpm.prompt_transforms[tid](raw_prompts)
        return transformed.mean(dim=0)  # (feature_dim,)

    def _select_prompts(
        self,
        mpm: Any,
        image_feat: Optional[torch.Tensor],
        text_feat: torch.Tensor,
        training: bool,
    ) -> None:
        """Compute guidance coefficients and set selected_prompt_indices on the PEFT tuner."""
        bs = text_feat.shape[0]
        tid = int(self.cur_task)  # 0-indexed current task

        if training:
            # Cosine similarity loss for current task's prototype
            cur_proto = self._compute_current_prototype(mpm)
            if image_feat is not None:
                self._image_cos_sim_loss = (
                    torch.tensor(1.0, device=image_feat.device, dtype=image_feat.dtype)
                    - F.cosine_similarity(image_feat, repeat(cur_proto, "c -> b c", b=bs), dim=-1).mean()
                )
            else:
                self._image_cos_sim_loss = torch.tensor(0.0, device=text_feat.device)
            self._text_cos_sim_loss = (
                torch.tensor(1.0, device=text_feat.device, dtype=text_feat.dtype)
                - F.cosine_similarity(text_feat, repeat(cur_proto, "c -> b c", b=bs), dim=-1).mean()
            )

            if self.cur_task == 0:
                # First task: only current task's prompt
                mpm.selected_prompt_indices = [tid]
                return

            # Compute prototypes from COMPLETED (past) tasks
            proto_embeddings = self._compute_prototypes(mpm)  # (cur_task, feature_dim)

            # Dual-modal guidance coefficients for past tasks
            guide_coef = torch.zeros(bs, self.cur_task, device=text_feat.device)
            if image_feat is not None:
                img_sim = F.cosine_similarity(
                    repeat(image_feat, "b c -> b n c", n=self.cur_task),
                    repeat(proto_embeddings, "n c -> b n c", b=bs),
                    dim=-1,
                )
                guide_coef += self.lam * img_sim

            txt_sim = F.cosine_similarity(
                repeat(text_feat, "b c -> b n c", n=self.cur_task),
                repeat(proto_embeddings, "n c -> b n c", b=bs),
                dim=-1,
            )
            guide_coef += (1.0 - self.lam) * txt_sim

            if self.cur_task < self.transfer_num:
                # Use all past prompts + current
                mpm.selected_prompt_indices = list(range(self.cur_task + 1))
            else:
                # Top-K from past tasks (excluding current) + current
                select_topk = self.transfer_num - 1
                past_scores = guide_coef[0, :tid]
                top_k_past = torch.topk(past_scores, k=min(select_topk, past_scores.shape[0]), dim=-1)[1]
                top_k_past = torch.flip(top_k_past, dims=[0])  # most relevant last (closest to input)
                mpm.selected_prompt_indices = top_k_past.tolist() + [tid]
        else:
            # Inference: select top-K from completed tasks
            if self.cur_task == 0:
                mpm.selected_prompt_indices = [0]
                return
            proto_embeddings = self._compute_prototypes(mpm)
            guide_coef = torch.zeros(bs, self.cur_task, device=text_feat.device)
            if image_feat is not None:
                img_sim = F.cosine_similarity(
                    repeat(image_feat, "b c -> b n c", n=self.cur_task),
                    repeat(proto_embeddings, "n c -> b n c", b=bs),
                    dim=-1,
                )
                guide_coef += self.lam * img_sim
            txt_sim = F.cosine_similarity(
                repeat(text_feat, "b c -> b n c", n=self.cur_task),
                repeat(proto_embeddings, "n c -> b n c", b=bs),
                dim=-1,
            )
            guide_coef += (1.0 - self.lam) * txt_sim
            top_k = min(self.transfer_num, self.cur_task)
            top_guide = torch.topk(guide_coef[0], k=top_k, dim=-1)[1]
            top_guide = torch.flip(top_guide, dims=[0])
            mpm.selected_prompt_indices = top_guide.tolist()

    # ------------------------------------------------------------------
    # CLIntegration interface
    # ------------------------------------------------------------------

    def initialize_model(self, model) -> None:
        self._model_ref = model
        device = next(model.parameters()).device

        # Freeze backbone
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_modal_prompt_peft(model)

        # After PEFT setup, sync trainability
        self._sync_trainability(model)

    def _setup_modal_prompt_peft(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.modal_prompt import ModalPromptConfig

        _base = getattr(model, "_base_model", None) or model

        mp_cfg = ModalPromptConfig(
            num_tasks=self.task_num,
            prefix_len=self.prefix_len,
            feature_dim=self.feature_dim,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            inference_mode=False,
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        if getattr(model, "_base_model", None) is not None:
            peft_model = get_peft_model(_base, mp_cfg)
            object.__setattr__(model, "_base_model", peft_model)
            model._modules["_base_model"] = peft_model
        else:
            get_peft_model(model, mp_cfg)

        peft_wrapped = getattr(model, "_base_model", None)
        if peft_wrapped is not None and hasattr(peft_wrapped, "print_trainable_parameters"):
            peft_wrapped.print_trainable_parameters()

    def _sync_trainability(self, model) -> None:
        mpm = self._find_modal_prompt_model(model)
        if mpm is None:
            return
        if model.training:
            tid = int(getattr(self.config, "cur_task", self.cur_task))
            mpm.set_trainable_prompts(tid)
            mpm.set_trainable_transforms(tid)
        else:
            mpm.set_trainable_prompts(None)
            mpm.set_trainable_transforms(None)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return

        # Update cur_task from config (may have changed between tasks)
        self.cur_task = int(getattr(self.config, "cur_task", self.cur_task))

        mpm = self._find_modal_prompt_model(model)
        if mpm is None:
            return

        image_feat, text_feat = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )

        self._select_prompts(mpm, image_feat, text_feat, training=model.training)

    def on_forward_start(self, model, context: CLContext) -> None:
        self._sync_trainability(model)

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            self._image_cos_sim_loss = None
            self._text_cos_sim_loss = None
        return outputs

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        self.cur_task = int(task_id) + 1

    def pre_generate_hook(self, model, input_ids, images, context: CLContext) -> bool:
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return True

        mpm = self._find_modal_prompt_model(model)
        if mpm is None:
            return True

        # Use the config's cur_task for inference routing
        infer_cur_task = int(getattr(self.config, "cur_task", self.cur_task))
        saved_cur = self.cur_task
        self.cur_task = max(infer_cur_task, 1)

        if images is None:
            mpm.selected_prompt_indices = [self.cur_task - 1]
            return True

        image_feat, text_feat = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )
        self._select_prompts(mpm, image_feat, text_feat, training=False)
        self.cur_task = saved_cur
        return True

    def get_inference_config(self) -> Dict:
        return {
            "task_num": self.task_num,
            "prefix_len": self.prefix_len,
            "transfer_num": self.transfer_num,
            "feature_dim": self.feature_dim,
            "lam": self.lam,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        state: Dict[str, Any] = {}
        state["task_num"] = self.task_num
        state["prefix_len"] = self.prefix_len
        state["transfer_num"] = self.transfer_num
        state["feature_dim"] = self.feature_dim
        state["lam"] = self.lam
        state["cur_task"] = self.cur_task

        root = model if model is not None else self._model_ref
        mpm = self._find_modal_prompt_model(root) if root is not None else None
        if mpm is not None:
            state["task_prompts"] = [p.detach().cpu().clone() for p in mpm.task_prompts]
            state["prompt_transforms"] = [
                {k: v.detach().cpu().clone() for k, v in t.state_dict().items()}
                for t in mpm.prompt_transforms
            ]

        torch.save(state, os.path.join(output_dir, "modal_prompt_state.pt"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        path = os.path.join(load_dir, "modal_prompt_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return False

        # Restore config
        for key in ("task_num", "prefix_len", "transfer_num", "feature_dim", "lam", "cur_task"):
            if key in state:
                setattr(self, key, state[key])

        if model is not None:
            mpm = self._find_modal_prompt_model(model)
            if mpm is None:
                pass
            else:
                if "task_prompts" in state:
                    blobs = state["task_prompts"]
                    n = min(len(blobs), len(mpm.task_prompts))
                    for i in range(n):
                        t = blobs[i]
                        if not isinstance(t, torch.Tensor):
                            continue
                        if t.shape != mpm.task_prompts[i].shape:
                            continue
                        mpm.task_prompts[i].data.copy_(
                            t.to(device=mpm.task_prompts[i].device, dtype=mpm.task_prompts[i].dtype)
                        )
                if "prompt_transforms" in state:
                    blobs = state["prompt_transforms"]
                    n = min(len(blobs), len(mpm.prompt_transforms))
                    for i in range(n):
                        sd = blobs[i]
                        if not isinstance(sd, dict):
                            continue
                        mpm.prompt_transforms[i].load_state_dict(sd, strict=False)

        return True
