from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION


# TRAIN_FLAG_OVERRIDES = {
#     "--method": "olora",
#     "--mm_projector_lr": "1e-3",
#     "--num_train_epochs": "1",
#     "--learning_rate": "1e-03",
#     "--warmup_ratio": "0",
#     "--lr_scheduler_type": "constant",
#     "--logging_steps": "1",
#     "--model_max_length": "2048",
#     "--dataloader_num_workers": "4",
# }

TRAIN_FLAG_OVERRIDES = {
    "--method": "olora",
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
    "batch_size": 12,
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
        0: 6,
        1: 4,
        2: 4,
        3: 4,
        4: 4,
        5: 4,
    },
}

METHOD_CONFIG = {
    # Reference run_uie_lora.py uses lora_dropout=0.1
    "lora_dropout": 0.1,
    "peft_target_modules": "attn_and_ffn",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
    "olora_lambda": 0.5,
    # Log CE / orth / total every N forwards during training; 0 disables
    "olora_orthogonal_log_interval": 50,
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