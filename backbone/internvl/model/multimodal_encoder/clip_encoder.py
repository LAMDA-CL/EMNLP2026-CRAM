import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionConfig
from .intern_vit_6b.configuration_intern_vit import InternVisionConfig
from .intern_vit_6b.modeling_intern_vit import InternVisionModel


from .intern_utils import (
    build_intern_vit_image_processor,
    intern_vit_image_size,
    is_intern_vit_6b_model,
    pretrained_load_kwargs,
)


class MLLMVisionTower(nn.Module):
    """InternViT-6B encoder for MLLM patch features (not used for routing)."""

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.load_separately = getattr(args, "load_vision_tower_separately", True)

        if self.load_separately:
            if not delay_load:
                self.load_model()
            elif is_intern_vit_6b_model(self.vision_tower_name):
                self.cfg_only = InternVisionConfig.from_pretrained(
                    self.vision_tower_name, **pretrained_load_kwargs(self.vision_tower_name)
                )
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(
                    self.vision_tower_name, **pretrained_load_kwargs(self.vision_tower_name)
                )
        else:
            self._init_vision_structure_for_checkpoint()

    def _init_vision_structure_for_checkpoint(self):
        """Build InternViT module tree so parent from_pretrained can load vision weights."""
        if not is_intern_vit_6b_model(self.vision_tower_name):
            raise ValueError(f"InternVL backbone expects InternViT-6B vision tower, got {self.vision_tower_name!r}")
        load_kw = pretrained_load_kwargs(self.vision_tower_name)
        cfg = InternVisionConfig.from_pretrained(self.vision_tower_name, **load_kw)
        embedded = not self.load_separately
        if embedded:
            from config.backbone.registry import get_embedded_vision_image_size

            cfg.image_size = get_embedded_vision_image_size("internvl")
        self.image_processor = build_intern_vit_image_processor(
            self.vision_tower_name,
            config=cfg,
            embedded_in_chat=embedded,
        )
        self.vision_tower = InternVisionModel(cfg)
        self.vision_tower.requires_grad_(False)

    def mark_loaded_from_checkpoint(self):
        if not self.is_loaded:
            if not hasattr(self, "vision_tower") or self.vision_tower is None:
                self._init_vision_structure_for_checkpoint()
            self.is_loaded = True

    def load_model(self):
        if not is_intern_vit_6b_model(self.vision_tower_name):
            raise ValueError(f"InternVL backbone expects InternViT-6B vision tower, got {self.vision_tower_name!r}")
        load_kw = pretrained_load_kwargs(self.vision_tower_name)
        self.vision_tower = InternVisionModel.from_pretrained(self.vision_tower_name, **load_kw)
        self.vision_tower.requires_grad_(False)
        self.image_processor = build_intern_vit_image_processor(
            self.vision_tower_name,
            config=self.vision_tower.config,
            embedded_in_chat=False,
        )
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_features.append(self.feature_select(image_forward_out).to(image.dtype))
            return image_features
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if hasattr(self, "vision_tower") and self.vision_tower is not None:
            return self.vision_tower.config
        if hasattr(self, "cfg_only"):
            return self.cfg_only
        raise RuntimeError("Vision tower config is not available before load/init.")

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        side = intern_vit_image_size(
            self.vision_tower_name,
            config=self.config,
            embedded_in_chat=not self.load_separately,
        )
        patch = getattr(self.config, "patch_size", 14)
        return (side // patch) ** 2


# Backward-compatible alias used by builder
CLIPVisionTower = MLLMVisionTower


class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, delay_load=False):
        super().__init__()
        from transformers import CLIPTextConfig, CLIPTextModel

        self.is_loaded = False
        self.text_tower_name = text_tower
        self.select_layer = args.mm_text_select_layer
        if not delay_load:
            self.load_model()
        else:
            load_kw = pretrained_load_kwargs(self.text_tower_name)
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_tower_name, **load_kw)

    def load_model(self):
        from transformers import CLIPTextModel

        load_kw = pretrained_load_kwargs(self.text_tower_name)
        self.text_tower = CLIPTextModel.from_pretrained(self.text_tower_name, **load_kw)
        self.text_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, text_inputs, return_hidden_states=False):
        text_forward_outs = self.text_tower(**(text_inputs.to(self.device)))
        if return_hidden_states:
            text_hidden_features = text_forward_outs.last_hidden_state.to(self.dtype)
            text_features = text_forward_outs.pooler_output.to(self.dtype)
            return [text_hidden_features, text_features]
        return text_forward_outs.pooler_output.to(self.dtype)

    @property
    def dtype(self):
        return self.text_tower.dtype

    @property
    def device(self):
        return self.text_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
