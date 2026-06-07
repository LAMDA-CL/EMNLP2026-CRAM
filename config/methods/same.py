"""
Defaults for method: same

Optional ``METHOD_CONFIG["exclude_module_path_segments"]`` scopes PEFT paths; default keeps injection on the LLM only.

``peft_target_modules``: this method defaults to **FFN only** (``gate_proj`` / ``up_proj`` / ``down_proj``); override via ``METHOD_CONFIG`` or ``--peft_target_modules`` (see ``PEFT/utils/peft_target_modules.py``).
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "same",
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

from utils.infer_paths import DEFAULT_ROUTING_VISION_TOWER_PATH, DEFAULT_TEXT_TOWER_PATH

INFER_DEFAULTS = {
    # Batch size for `backbone.shared.eval.model_unified` (InferenceEngine)
    "batch_size": 1,
    "routing_vision_tower": DEFAULT_ROUTING_VISION_TOWER_PATH,
    "text_tower": DEFAULT_TEXT_TOWER_PATH,
}

# Gradient accumulation: int, or {benchmark: int|{task_id: int}}.
# ``-1`` → global_batch // (per_device_batch × num_gpus); global batch from
# ``TRAIN_GLOBAL_BATCH_SIZE`` or benchmark task ``batch_size``.
# TRAIN_GRADIENT_ACCUMULATION_STEPS = -1

# Optional target global effective batch (per_device × gpus × grad_acc).
# When omitted and grad_acc=-1, falls back to benchmark task ``batch_size``.
# TRAIN_GLOBAL_BATCH_SIZE = {
#     "ucit": 24,
#     "coin": 96,
# }

# Keys = task index in config/benchmarks/* (``run.py train <id>``). CoIN: 0–7, UCIT: 0–5.
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
        1: 6,
        2: 6,
        3: 6,
        4: 6,
        5: 6,
    },
    "trigap": {
        0: 12,
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 8,
        9: 8,
    },
}

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "ffn",
    # SAME spectral / curvature knobs (see PEFT.tuners.custom.same.SAMELinear)
    "curvature_mu": 0.9,
    "window_size": 3,
    "max_components": 64,
    # Cumulative singular-value energy threshold for principal directions (legacy default 0.9)
    "cumulative_energy_ratio": 0.9,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}

# Overlay after METHOD_CONFIG, before METHOD_CONFIG_BY_BENCHMARK (see merge_method_config_into).
METHOD_CONFIG_BY_BACKBONE = {
    "llava": {
        "tau_score": 0.1,
    },
    "internvl": {
        "tau_score": 0.1,
    },
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