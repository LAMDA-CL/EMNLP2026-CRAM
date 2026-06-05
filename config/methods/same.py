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
    "--gradient_accumulation_steps": "1",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    # Batch size for `backbone.shared.eval.model_unified` (InferenceEngine)
    "batch_size": 1,
}

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
        0: 12,
        1: 12,
        2: 12,
        3: 12,
        4: 12,
        5: 12,
    },
    "trigap": {
        0: 12,
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
    },
}

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "ffn",
    # SAME spectral / curvature knobs (see PEFT.tuners.custom.same.SAMELinear)
    "tau_score": 0.1,
    "curvature_mu": 0.9,
    "window_size": 3,
    "max_components": 64,
    # Cumulative singular-value energy threshold for principal directions (legacy default 0.9)
    "cumulative_energy_ratio": 0.9,
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