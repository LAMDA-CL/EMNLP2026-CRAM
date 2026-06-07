"""
replay_lora: standard LoRA (default attention+FFN) + task-partitioned experience replay.

Replay hyperparameters in ``METHOD_CONFIG``:

- ``replay_buffer_size``: total capacity target (slots split across ``task_num-1`` tasks).
- ``replay_sample_prob``: per-sample Bernoulli on each **current-batch** row (``cl_raw_example``) for buffer writes;
  override ``Replay_loraIntegration.should_store_training_example`` for custom policies (see ``method/base/integration.py``).

``peft_target_modules`` defaults to ``attn_and_ffn`` (see ``PEFT/utils/peft_target_modules.py``).
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "replay_lora",
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
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
    },
    "ucit": {i: 6 for i in range(6)},
}

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "attn",
    "exclude_module_path_segments": EXCLUDE_FOR_LLM_ONLY_INJECTION,
    "replay_buffer_size": 180,
    "replay_sample_prob": 0.7,
}

#we reduce the rank as it introduces extra samples which should be stored.
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