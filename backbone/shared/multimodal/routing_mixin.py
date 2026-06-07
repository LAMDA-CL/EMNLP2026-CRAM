"""Shared routing vision tower initialization (CLIP, separate from MLLM vision encoder)."""
from __future__ import annotations

from backbone.shared.multimodal_encoder import RoutingVisionTower


class RoutingVisionMixin:
    def get_routing_vision_tower(self):
        tower = getattr(self, "routing_vision_tower", None)
        if type(tower) is list:
            tower = tower[0]
        return tower

    def initialize_routing_vision_modules(self, model_args, fsdp=None) -> None:
        routing_path = getattr(model_args, "routing_vision_tower", None)
        if not routing_path:
            self.routing_vision_tower = None
            return

        if self.get_routing_vision_tower() is None:
            tower = RoutingVisionTower(routing_path, delay_load=False)
            if fsdp is not None and len(fsdp) > 0:
                self.routing_vision_tower = [tower]
            else:
                self.routing_vision_tower = tower
        else:
            tower = self.get_routing_vision_tower()
            if not tower.is_loaded:
                tower.load_model()

        self.config.routing_vision_tower = routing_path
