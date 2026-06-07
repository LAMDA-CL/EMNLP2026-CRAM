"""Shared CLIP vision tower for routing (768-d image_embeds), separate from MLLM vision encoder."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection


def _pretrained_load_kwargs(path: str) -> dict:
    if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return {"local_files_only": True}
    return {}


class RoutingVisionTower(nn.Module):
    """Frozen CLIP vision tower used only for routing / anchor features."""

    def __init__(self, vision_tower_path: str, delay_load: bool = False):
        super().__init__()
        self.vision_tower_path = vision_tower_path
        self.is_loaded = False
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                self.vision_tower_path, **_pretrained_load_kwargs(self.vision_tower_path)
            )

    def load_model(self) -> None:
        load_kw = _pretrained_load_kwargs(self.vision_tower_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path, **load_kw)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_path, **load_kw)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if type(images) is list:
            feats = []
            for image in images:
                out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=False,
                )
                feats.append(out.image_embeds.to(image.dtype))
            return torch.cat(feats, dim=0)
        out = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=False)
        return out.image_embeds.to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self) -> int:
        return int(self.config.projection_dim)
