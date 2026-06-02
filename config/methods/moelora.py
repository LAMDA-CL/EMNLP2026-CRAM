"""
Defaults for method: moelora
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "moelora",
    "--freeze_mm_mlp_adapter": "True",
    "--mm_projector_lr": "2e-5",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    "batch_size": 1,
}

TRAIN_BATCH_SIZES = {
    "coin": {
        0: 6,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
        5: 6,
        6: 6,
        7: 6,
    },
    "ucit": {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
    },
}

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "attn_and_ffn",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}

METHOD_CONFIG_BY_BENCHMARK = {
    "coin": {
        "lora_r": 64,
        "lora_alpha": 128,
    },
    "ucit": {
        "lora_r": 96,
        "lora_alpha": 192,
    },
    "trigap": {
        "lora_r": 80,
        "lora_alpha": 160,
    },
}