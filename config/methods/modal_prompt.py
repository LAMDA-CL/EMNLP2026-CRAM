"""
Defaults for method: modal_prompt

ModalPrompt: per-task soft prompts + prompt transform MLPs with
dual-modal (image + text) guided top-K prompt selection.
"""
from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION


TRAIN_FLAG_OVERRIDES = {
    "--method": "modal_prompt",
    "--freeze_mm_mlp_adapter": "True",
    "--num_train_epochs": "10",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
    "--lora_enable": "False",
}

TRAIN_EXTRA_ARGS: list[str] = []

from utils.infer_paths import DEFAULT_ROUTING_VISION_TOWER_PATH, DEFAULT_TEXT_TOWER_PATH

INFER_DEFAULTS = {
    "clmethod": "modal_prompt",
    "batch_size": 1,
    "routing_vision_tower": DEFAULT_ROUTING_VISION_TOWER_PATH,
    "text_tower": DEFAULT_TEXT_TOWER_PATH,
}

TRAIN_BATCH_SIZES = {
    "coin": {i: 12 for i in range(8)},
    "ucit": {i: 4 for i in range(6)},
}

METHOD_CONFIG = {
    # Prompt configuration
    "prefix_len": 96,
    "cur_task": 0,

    # Modal guidance
    "clip_feature_dim": 768,
    "transfer_num": 1,
    "lam": 0.5,  # weight for image vs text guidance: lam*image + (1-lam)*text

    # PEFT injection scope (only LLM, exclude vision/text towers)
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
