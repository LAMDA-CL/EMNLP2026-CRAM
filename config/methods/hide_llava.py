"""
Defaults for method: hide_llava

PEFT path filtering: ``METHOD_CONFIG`` here pins injection to the LLM trunk only
(``PEFT.utils.peft_scope_defaults``). Use ``exclude_module_path_segments=[]`` for full-model injection;
custom skip lists are documented in ``config/methods/README.md``.

Layers: default **attention + FFN** inside the Transformer (``peft_target_modules: attn_and_ffn``), no ``lm_head``;
override with ``--peft_target_modules`` (see ``PEFT/utils/peft_target_modules.py``).
Per-benchmark LoRA sizes live in ``METHOD_CONFIG`` + ``METHOD_CONFIG_BY_BENCHMARK``; ``load_model_for_train`` copies
``lora_r`` / ``lora_alpha`` / ``lora_dropout`` into ``TrainingArguments`` (CLI ``--lora_*`` still wins).
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

# These are merged on top of config/run_config.py

TRAIN_FLAG_OVERRIDES = {
    "--method": "hide_llava",
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

# Keys = task index in config/benchmarks/* (same as ``run.py`` / ``train <id>``), not ``cur_task`` alias.
# CoIN: 8 tasks (0–7). UCIT: 6 tasks (0–5).


TRAIN_GRADIENT_ACCUMULATION_STEPS = 1

TRAIN_BATCH_SIZES = {
    "coin": {
        0: 12,
        1: 12,
        2: 12,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
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

# Method parameters (used by method factory / integrations if needed)
METHOD_CONFIG = {
    # Default LoRA size when benchmark not listed in METHOD_CONFIG_BY_BENCHMARK
    "lora_dropout": 0.05,
    # Attention ∪ FFN (no lm_head); matches HiDe PEFT type MOE_LORA_HiDe
    "peft_target_modules": "attn_and_ffn",
    # Explicit LLM-only injection (same idea as PEFT default None, kept for clarity)
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}

# METHOD_CONFIG overrides per benchmark (keep inference merges aligned with train)
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