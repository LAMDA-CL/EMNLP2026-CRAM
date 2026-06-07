from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from method.base.context import CLContext
from method.base.hooks import HookManager
from method.factory import CLMethodFactory
from method.base.routing_utils import extract_routing_image_features
from backbone.shared.peft_llm_targets import should_skip_module_for_peft_scan
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, List
import numpy as np
import os


_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft_model import PeftModelForCausalLMLORAMOEDisCo
    from PEFT.tuners.custom.disco import DiscoMOELoraConfig, DiscoMOELoraModel

    register_peft_extension(
        peft_type="MOE_LORA_DisCo",
        config_cls=DiscoMOELoraConfig,
        tuner_model_cls=DiscoMOELoraModel,
        task_type="CAUSAL_LM_DisCo",
        task_peft_model_cls=PeftModelForCausalLMLORAMOEDisCo,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("disco")
class DiscoIntegration(CLIntegration):

    def __init__(self, config: Any):
        super().__init__(config)
        self.hook_manager = HookManager()

        self.task_num = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim = int(getattr(config, "clip_feature_dim", 768))
        self.routing_temperature = float(getattr(config, "routing_temperature", 0.05))

        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None

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

        self._setup_disco_lora(model)

        for name, param in model.named_parameters():
            if any(p in name for p in ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']):
                param.requires_grad = False

    def _setup_disco_lora(self, model):
        try:
            ensure_peft_extension_registered()
            from PEFT.tuners.custom.disco import DiscoMOELoraConfig
            from PEFT import get_peft_model

            target_modules = self._find_target_modules(model)

            raw_r = int(getattr(self.config, 'lora_r', 64))
            if raw_r % self.task_num != 0:
                adjusted_r = ((raw_r // self.task_num) + 1) * self.task_num
            else:
                adjusted_r = raw_r

            lora_alpha = adjusted_r * 2

            lora_config = DiscoMOELoraConfig(
                target_modules=target_modules,
                r=adjusted_r,
                lora_alpha=lora_alpha,
                lora_dropout=float(getattr(self.config, 'lora_dropout', 0.05)),
                expert_num=self.task_num,
                cur_task=int(getattr(self.config, 'cur_task', 0)),
                task_type="CAUSAL_LM_DisCo",
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
            raise RuntimeError("DiscoMOELoraConfig not available") from e
        except Exception as e:
            raise RuntimeError("Disco LoRA setup failed") from e

    def _find_target_modules(self, model) -> List[str]:
        target_modules = set()
        _base_model = getattr(model, '_base_model', None) or model
        for name, module in _base_model.named_modules():
            if should_skip_module_for_peft_scan(name, self.config):
                continue
            if isinstance(module, torch.nn.Linear):
                if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                            'gate_proj', 'up_proj', 'down_proj']):
                    module_type = name.split('.')[-1]
                    target_modules.add(module_type)
        return list(target_modules)

    def on_input_prep(self, model, args, kwargs, context: CLContext):
        images = kwargs.get('images', None)
        input_ids = args[0] if args else None

        clip_tokenizer, text_tower = self._resolve_clip_components(model)
        if clip_tokenizer is None or text_tower is None:
            return

        if images is None:
            if model.training:
                self._set_lora_id_on_layers(model, context.task_id or 0)
            else:
                self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return

        image_feat, text_feat = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )

        if model.training:
            self._update_prototypes(image_feat, text_feat, context.task_id)
            self._set_lora_id_on_layers(model, context.task_id or 0)
        else:
            cos_sim_list = self._compute_soft_routing(image_feat, text_feat)
            self._propagate_mask_signal(model, cos_sim_list)

    def _resolve_clip_components(self, model):
        clip_tokenizer = getattr(model, 'clip_tokenizer', None)
        text_tower = getattr(model, 'text_tower', None)

        _base_model = getattr(model, '_base_model', None)
        if _base_model is not None:
            if clip_tokenizer is None:
                clip_tokenizer = getattr(_base_model, 'clip_tokenizer', None)
            if text_tower is None:
                text_tower = getattr(_base_model, 'text_tower', None)
        if _base_model is not None and (clip_tokenizer is None or text_tower is None):
            if hasattr(_base_model, 'base_model'):
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model.base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model.base_model, 'text_tower', None)
        return clip_tokenizer, text_tower

    def _extract_clip_features(self, model, images, input_ids, clip_tokenizer, text_tower):
        device = images.device

        try:
            image_guide_features = extract_routing_image_features(model, images)
        except RuntimeError:
            image_guide_features = torch.zeros(images.shape[0], self.feature_dim, device=device)

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
                decoded_clip, padding="longest", max_length=77, truncation=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                text_guide_features = text_tower(clip_inputs)
                if isinstance(text_guide_features, tuple):
                    text_guide_features = text_guide_features[0]
        else:
            text_guide_features = torch.zeros(1, self.feature_dim, device=device)

        if text_guide_features.dim() == 1:
            text_guide_features = text_guide_features.unsqueeze(0)

        return image_guide_features, text_guide_features

    def _update_prototypes(self, image_feat, text_feat, task_id):
        if task_id is None or task_id >= self.task_num:
            return
        batch_size = image_feat.shape[0]
        if batch_size == 0:
            return

        task_idx = int(task_id)
        with torch.no_grad():
            old_img_anchor = self.image_anchors[task_idx].data.clone()
            old_img_count = self.image_boundary[task_idx].data.clone()
            image_sum = old_img_anchor * old_img_count + image_feat.sum(dim=0)
            new_img_count = old_img_count + batch_size
            self.image_anchors[task_idx].data.copy_(image_sum / new_img_count)
            self.image_boundary[task_idx].data.copy_(new_img_count)
            self.image_anchors[task_idx].requires_grad = False
            self.image_boundary[task_idx].requires_grad = False

            old_txt_anchor = self.text_anchors[task_idx].data.clone()
            old_txt_count = self.text_boundary[task_idx].data.clone()
            text_sum = old_txt_anchor * old_txt_count + text_feat.sum(dim=0)
            new_txt_count = old_txt_count + batch_size
            self.text_anchors[task_idx].data.copy_(text_sum / new_txt_count)
            self.text_boundary[task_idx].data.copy_(new_txt_count)
            self.text_anchors[task_idx].requires_grad = False
            self.text_boundary[task_idx].requires_grad = False

    def _compute_soft_routing(self, image_feat, text_feat):
        device = text_feat.device
        text_feat_mean = text_feat.mean(dim=0, keepdim=True)

        global_text = torch.stack([p.to(device) for p in self.text_anchors]).squeeze(1)
        cos_sim = F.cosine_similarity(global_text, text_feat_mean, dim=1)
        cos_sim = cos_sim[:self.task_num]

        cos_sim_softmax = F.softmax(cos_sim / self.routing_temperature, dim=0)

        return cos_sim_softmax.tolist()

    def _propagate_mask_signal(self, model, cos_sim_list):
        for module in model.modules():
            if hasattr(module, "mask_signal"):
                module.mask_signal = cos_sim_list

    def _set_lora_id_on_layers(self, model, task_id):
        for module in model.modules():
            if hasattr(module, "lora_id"):
                module.lora_id = task_id

    def _text_only_routing(self, model, input_ids, clip_tokenizer, text_tower):
        if input_ids is None or input_ids.shape[1] <= 1:
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
            decoded_clip, padding="longest", max_length=77, truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_feat = text_tower(clip_inputs)
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        text_feat_mean = text_feat.mean(dim=0, keepdim=True)

        global_text = torch.stack([p.to(device) for p in self.text_anchors]).squeeze(1)
        cos_sim = F.cosine_similarity(global_text, text_feat_mean, dim=1)
        cos_sim = cos_sim[:self.task_num]
        cos_sim_softmax = F.softmax(cos_sim / self.routing_temperature, dim=0)

        sim_list = cos_sim_softmax.tolist()
        self._propagate_mask_signal(model, sim_list)

    def on_forward_start(self, model, context: CLContext):
        pass

    def on_forward_end(self, model, outputs, context: CLContext):
        return outputs

    def on_task_end(self, model, context: CLContext, task_id: int):
        pass

    def get_inference_config(self) -> Dict:
        return {
            "task_num": self.task_num,
            "feature_dim": self.feature_dim,
            "routing_temperature": self.routing_temperature,
        }

    def save_extra_state(self, output_dir: str, model=None) -> bool:
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

        if state:
            save_path = os.path.join(output_dir, 'disco_state.pt')
            torch.save(state, save_path)
            return True
        return False

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        load_path = os.path.join(load_dir, 'disco_state.pt')
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
        if images is None:
            task_id = context.task_id if context and context.task_id is not None else 0
            self._set_lora_id_on_layers(model, task_id)
            return True

        clip_tokenizer, text_tower = self._resolve_clip_components(model)
        if clip_tokenizer is None or text_tower is None:
            self._set_lora_id_on_layers(model, 0)
            return True

        image_feat, text_feat = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )
        cos_sim_list = self._compute_soft_routing(image_feat, text_feat)
        self._propagate_mask_signal(model, cos_sim_list)
        return True
