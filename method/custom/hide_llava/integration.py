from config.backbone.registry import get_routing_feature_dim
from method.base.routing_utils import extract_routing_image_features
from method.base.integration import CLIntegration
from method.base.context import CLContext
from method.base.hooks import HookManager
from method.factory import CLMethodFactory
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os


_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft_model import PeftModelForCausalLMLORAMOE
    from PEFT.tuners.custom.hidellava import HiDeMOELoraConfig, HiDeMOELoraModel

    register_peft_extension(
        peft_type="MOE_LORA_HiDe",
        config_cls=HiDeMOELoraConfig,
        tuner_model_cls=HiDeMOELoraModel,
        task_type="CAUSAL_LM_HiDe",
        task_peft_model_cls=PeftModelForCausalLMLORAMOE,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("hide_llava", "hide")
class Hide_llavaIntegration(CLIntegration):

    def __init__(self, config: Any):
        super().__init__(config)
        self.hook_manager = HookManager()

        self.task_num = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim = int(get_routing_feature_dim(getattr(config, "backbone", None)))

        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None

        self._last_predicted_task_id: Optional[int] = None

    def initialize_model(self, model):
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            param.requires_grad = False

        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)

        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)

        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)

        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.task_num = self.task_num

        self._setup_hide_lora(model)

        for name, param in model.named_parameters():
            if any(pattern in name for pattern in ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']):
                param.requires_grad = False

    def _setup_hide_lora(self, model):
        try:
            ensure_peft_extension_registered()
            from PEFT import HiDeMOELoraConfig, get_peft_model

            target_modules = self._find_target_modules(model)

            lora_config = HiDeMOELoraConfig(
                target_modules=target_modules,
                r=getattr(self.config, 'lora_r', 64),
                lora_alpha=getattr(self.config, 'lora_alpha', 128),
                lora_dropout=getattr(self.config, 'lora_dropout', 0.05),
                expert_num=self.task_num,
                cur_task=getattr(self.config, 'cur_task', 0),
                task_type="CAUSAL_LM",
                exclude_module_path_segments=self.peft_exclude_module_path_segments,
            )

            _base_model = getattr(model, '_base_model', None)
            if _base_model is not None:
                peft_model = get_peft_model(_base_model, lora_config)
                object.__setattr__(model, '_base_model', peft_model)
            else:
                peft_model = get_peft_model(model, lora_config)

            if hasattr(peft_model, 'print_trainable_parameters'):
                peft_model.print_trainable_parameters()

        except ImportError as e:
            raise RuntimeError("HiDeMOELoraConfig not available") from e
        except Exception as e:
            raise RuntimeError("HiDe LoRA setup failed") from e

    def _find_target_modules(self, model) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def on_input_prep(self, model, args, kwargs, context: CLContext):

        images = kwargs.get('images', None)
        input_ids = args[0] if args else None

        clip_tokenizer = getattr(model, 'clip_tokenizer', None)
        text_tower = getattr(model, 'text_tower', None)

        if clip_tokenizer is None or text_tower is None:
            _base_model = getattr(model, '_base_model', None)
            if _base_model is not None:
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model, 'text_tower', None)

        if clip_tokenizer is None or text_tower is None:
            if _base_model is not None and hasattr(_base_model, 'base_model'):
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model.base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model.base_model, 'text_tower', None)

        if clip_tokenizer is None or text_tower is None:
            return

        if images is None:
            if model.training:
                tid = getattr(context, "task_id", None)
                if tid is not None:
                    try:
                        tid_i = int(tid)
                        if 0 <= tid_i < self.task_num:
                            self._propagate_task_id(model, tid_i)
                        else:
                            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
                    except (TypeError, ValueError):
                        self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
                else:
                    self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            else:
                self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return
        else:
            image_guide_features, text_guide_features = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)

            if model.training:
                self._update_prototypes(
                    image_guide_features, text_guide_features,
                    context.task_id, context,
                )
            else:
                predicted_task_id = self._predict_task(
                    image_guide_features, text_guide_features, context
                )
                self._propagate_task_id(model, predicted_task_id)


    def _extract_clip_features(
        self, model, images, input_ids, clip_tokenizer, text_tower
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = images.device

        image_guide_features = extract_routing_image_features(model, images)

        text_guide_features: Optional[torch.Tensor] = None

        if input_ids is not None and clip_tokenizer is not None:
            main_tokenizer = getattr(model, 'tokenizer', None)
            if main_tokenizer is None:
                _base_model = getattr(model, '_base_model', None)
                if _base_model is not None:
                    main_tokenizer = getattr(_base_model, 'tokenizer', None)

            if main_tokenizer is not None:
                input_pad = np.where(
                    input_ids.cpu().detach().numpy() != -200,
                    input_ids.cpu().detach().numpy(),
                    main_tokenizer.pad_token_id,
                )
                decoded_inputs = main_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
                decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
                decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]
            else:
                input_pad = np.where(
                    input_ids.cpu().detach().numpy() != -200,
                    input_ids.cpu().detach().numpy(),
                    clip_tokenizer.pad_token_id,
                )
                decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
                decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
                decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]

            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                text_guide_features = text_tower(clip_inputs)
                if isinstance(text_guide_features, tuple):
                    text_guide_features = text_guide_features[0]

        if text_guide_features is not None and text_guide_features.dim() == 1:
            text_guide_features = text_guide_features.unsqueeze(0)

        return image_guide_features, text_guide_features


    def _update_prototypes(
        self,
        image_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor],
        task_id: Optional[int],
        context: CLContext,
    ):
        if task_id is None or task_id >= self.task_num:
            return

        batch_size = image_feat.shape[0]
        if batch_size == 0:
            return

        task_idx = task_id

        with torch.no_grad():
            old_img_anchor = self.image_anchors[task_idx].data.clone()
            old_img_count = self.image_boundary[task_idx].data.clone()

            image_sum = old_img_anchor * old_img_count + image_feat.sum(dim=0)
            new_img_count = old_img_count + batch_size

            self.image_anchors[task_idx].data.copy_(image_sum / new_img_count)
            self.image_boundary[task_idx].data.copy_(new_img_count)

            self.image_anchors[task_idx].requires_grad = False
            self.image_boundary[task_idx].requires_grad = False

            if text_feat is not None:
                old_txt_anchor = self.text_anchors[task_idx].data.clone()
                old_txt_count = self.text_boundary[task_idx].data.clone()

                text_sum = old_txt_anchor * old_txt_count + text_feat.sum(dim=0)
                new_txt_count = old_txt_count + batch_size

                self.text_anchors[task_idx].data.copy_(text_sum / new_txt_count)
                self.text_boundary[task_idx].data.copy_(new_txt_count)

                self.text_anchors[task_idx].requires_grad = False
                self.text_boundary[task_idx].requires_grad = False

    def _task_cosine_max_vs_anchor(
        self, feat: torch.Tensor, anchors: torch.nn.ParameterList, t: int
    ) -> float:
        anchor = anchors[t].to(feat.device)
        return F.cosine_similarity(
            feat.unsqueeze(1),
            anchor.unsqueeze(0),
            dim=2,
        ).max().item()

    def _predict_task(
        self,
        image_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor],
        context: CLContext,
    ) -> int:
        device = image_feat.device

        image_sims = [
            self._task_cosine_max_vs_anchor(image_feat, self.image_anchors, t)
            for t in range(self.task_num)
        ]

        if text_feat is None:
            sim = torch.tensor(image_sims, device=device)
            predicted_task_id = int(torch.argmax(sim).item())
            return predicted_task_id

        text_sims = [
            self._task_cosine_max_vs_anchor(text_feat, self.text_anchors, t)
            for t in range(self.task_num)
        ]
        combined = [0.5 * img + 0.5 * txt for img, txt in zip(image_sims, text_sims)]
        sim = torch.tensor(combined, device=device)
        predicted_task_id = int(torch.argmax(sim).item())

        return predicted_task_id

    def _propagate_task_id(self, model, task_id: int):
        for module in model.modules():
            if hasattr(module, "predicted_task_id"):
                module.predicted_task_id = task_id

        model._last_predicted_task_id = task_id

    def _inference_routing_fallback_task_id(self, model) -> int:
        last = getattr(model, "_last_predicted_task_id", None)
        if last is not None and isinstance(last, int) and 0 <= last < self.task_num:
            return last
        return 0

    def _text_only_routing(self, model, input_ids, clip_tokenizer, text_tower):
        if input_ids is None or input_ids.shape[1] <= 1:
            if hasattr(model, '_last_predicted_task_id') and model._last_predicted_task_id is not None:
                self._propagate_task_id(model, model._last_predicted_task_id)
            return

        device = next(model.parameters()).device

        input_pad = np.where(
            input_ids.cpu().detach().numpy() != -200,
            input_ids.cpu().detach().numpy(),
            clip_tokenizer.pad_token_id,
        )
        decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
        decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]

        clip_inputs = clip_tokenizer(
            decoded_clip,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_feat = text_tower(clip_inputs)
            text_feat = text_feat.to(device)

        for t in range(self.task_num):
            if hasattr(self.text_anchors[t], 'to'):
                self.text_anchors[t] = self.text_anchors[t].to(device)

        text_sims = []
        for t in range(self.task_num):
            text_feat_device = text_feat.device
            anchor_device = self.text_anchors[t].device

            if text_feat_device != anchor_device:
                self.text_anchors[t] = self.text_anchors[t].to(text_feat_device)

            sim = F.cosine_similarity(
                text_feat.unsqueeze(1),
                self.text_anchors[t].unsqueeze(0),
                dim=2
            ).max().item()
            text_sims.append(sim)

        predicted_task_id = int(torch.argmax(torch.tensor(text_sims)).item())
        self._propagate_task_id(model, predicted_task_id)

    def on_forward_start(self,model, context: CLContext):
        pass

    def on_forward_end(self, model, outputs, context: CLContext):
        return outputs

    def on_step_end(self, model, context: CLContext, loss=None):
        pass

    def on_task_end(self, model, context: CLContext, task_id: int):
        pass

    def get_inference_config(self) -> Dict:
        return {
            "task_num": self.task_num,
            "feature_dim": self.feature_dim,
        }

    def save_extra_state(self, output_dir: str, model=None):
        import os
        import torch

        os.makedirs(output_dir, exist_ok=True)

        state = {}

        if self.image_anchors is not None:
            state['image_anchors'] = [p.cpu().clone() for p in self.image_anchors]

        if self.text_anchors is not None:
            state['text_anchors'] = [p.cpu().clone() for p in self.text_anchors]

        if self.image_boundary is not None:
            state['image_boundary'] = [p.cpu().clone() for p in self.image_boundary]

        if self.text_boundary is not None:
            state['text_boundary'] = [p.cpu().clone() for p in self.text_boundary]

        state["task_num"] = self.task_num
        state["expert_num"] = self.task_num
        state['_last_predicted_task_id'] = self._last_predicted_task_id

        if state:
            save_path = os.path.join(output_dir, 'hide_state.pt')
            torch.save(state, save_path)
            return True
        else:
            return False

    def load_extra_state(self, load_dir: str, model=None):
        import os
        import torch

        load_path = os.path.join(load_dir, 'hide_state.pt')
        if not os.path.exists(load_path):
            return False

        state = torch.load(load_path, map_location='cpu')

        if 'image_anchors' in state and self.image_anchors is not None:
            for i, p in enumerate(state['image_anchors']):
                if i < len(self.image_anchors):
                    self.image_anchors[i].data.copy_(p)

        if 'text_anchors' in state and self.text_anchors is not None:
            for i, p in enumerate(state['text_anchors']):
                if i < len(self.text_anchors):
                    self.text_anchors[i].data.copy_(p)

        if 'image_boundary' in state and self.image_boundary is not None:
            for i, p in enumerate(state['image_boundary']):
                if i < len(self.image_boundary):
                    self.image_boundary[i].data.copy_(p)
        if 'text_boundary' in state and self.text_boundary is not None:
            for i, p in enumerate(state['text_boundary']):
                if i < len(self.text_boundary):
                    self.text_boundary[i].data.copy_(p)

        if model is not None:
            if self.image_anchors is not None:
                object.__setattr__(model, 'image_anchors', self.image_anchors)
            if self.text_anchors is not None:
                object.__setattr__(model, 'text_anchors', self.text_anchors)
            if self.image_boundary is not None:
                object.__setattr__(model, 'image_boundary', self.image_boundary)
            if self.text_boundary is not None:
                object.__setattr__(model, 'text_boundary', self.text_boundary)

        return True

    def pre_generate_hook(self, model, input_ids, images, context) -> bool:
        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = getattr(model, "text_tower", None)
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            if clip_tokenizer is None:
                clip_tokenizer = getattr(_base_model, "clip_tokenizer", None)
            if text_tower is None:
                text_tower = getattr(_base_model, "text_tower", None)
        if _base_model is not None and hasattr(_base_model, "base_model"):
            if clip_tokenizer is None:
                clip_tokenizer = getattr(_base_model.base_model, "clip_tokenizer", None)
            if text_tower is None:
                text_tower = getattr(_base_model.base_model, "text_tower", None)

        if images is None:
            if clip_tokenizer is None or text_tower is None:
                fb = self._inference_routing_fallback_task_id(model)
                self._propagate_task_id(model, fb)
                return True
            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return True

        if clip_tokenizer is None or text_tower is None:
            fb = self._inference_routing_fallback_task_id(model)
            self._propagate_task_id(model, fb)
            return True

        image_guide_features, text_guide_features = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )
        predicted_task_id = self._predict_task(
            image_guide_features, text_guide_features, context
        )
        self._propagate_task_id(model, predicted_task_id)
        return True
