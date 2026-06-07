"""Backbone-agnostic HF Trainer with CL hooks and mm_projector LR groups."""
from backbone.shared.train.llava_trainer import MLLMTrainer, LLaVATrainer

__all__ = ["MLLMTrainer", "LLaVATrainer"]
