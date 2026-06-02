from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "ewc",
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
        0: 12,
        1: 12,
        2: 12,
        3: 12,
        4: 12,
        5: 12,
        6: 12,
        7: 12,
    },
    "ucit": {
        0: 8,
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 4,
    },
}

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "attn_and_ffn",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
    "ewc_lambda": 5000,
    "ewc_fisher_batches": 50,
    # Fisher accumulation runs at on_train_end (not tied to per-device batch); tune per GPU memory
    "ewc_fisher_micro_batch_size": 1,
    # Chunked EWC grad injection (elements per slice) to cap GPU temporaries; lower if still OOM.
    "ewc_penalty_chunk_elements": 16384,
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
