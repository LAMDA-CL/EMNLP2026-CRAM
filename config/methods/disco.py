"""
Defaults for method: disco

DISCO (ICLR 2025) — Continual Instruction Tuning for Multimodal LLMs.
Uses MoE-LoRA with soft routing via lora_AB diagonal mask + CLIP prototype matching.
"""
from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION


TRAIN_FLAG_OVERRIDES = {
    "--method": "disco",
    "--freeze_mm_mlp_adapter": "False",
    "--mm_projector_lr": "2e-5",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--weight_decay": "0.0",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
    "--gradient_accumulation_steps": "1",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    "clmethod": "disco",
    "batch_size": 1,
}

TRAIN_BATCH_SIZES = {
    "coin": {i: 8 for i in range(8)},
    "ucit": {i: 8 for i in range(6)},
}

METHOD_CONFIG = {
    "peft_target_modules": "ffn",
    "clip_feature_dim": 768,
    "cur_task": 0,
    "lora_r": 96,
    "lora_alpha": 192,
    "lora_dropout": 0.05,
    "routing_temperature": 0.05,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
METHOD_CONFIG_BY_BENCHMARK = {
    "coin": {
        "lora_r": 64,
        "lora_alpha": 128,
    },
    "ucit": {
        "lora_r": 48,
        "lora_alpha": 96,
    },
    "trigap": {
        "lora_r": 80,
        "lora_alpha": 160,
    },
}