"""
Defaults for method ``smolora`` (SMoLoRA).

Instruction side priority: ``ins_emb_path`` (custom .pkl) >
``smolora_builtin_sentence_ins_emb`` (built-in MiniLM, same role as ``ins_gen.py`` single) >
else CLIP ``text_tower`` (feature dim ``config.backbone.llava.CLIP_FEATURE_DIM``, typically 768).

PEFT placement: ``METHOD_CONFIG["peft_target_modules"]`` or ``--peft_target_modules`` (``attn`` / ``ffn`` / ``linear``, …); default ``attention``; see ``PEFT/utils/peft_target_modules.py``.
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "smolora",
    "--freeze_mm_mlp_adapter": "True",
    "--lora_r": "64",
    "--lora_alpha": "128",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--model_max_length": "2048",
    "--gradient_accumulation_steps": "1",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    "batch_size": 8,
}

TRAIN_BATCH_SIZES = {
    "coin": {i: 8 for i in range(8)},
    "ucit": {i: 8 for i in range(6)},
}

METHOD_CONFIG = {
    "smolora_expert_num": 8,
    "ins_emb_path": None,
    "smolora_builtin_sentence_ins_emb": True,
    "smolora_sentence_transformer_model": "/root/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",
    "smolora_clip_no_grad": True,
    "ins_type": 0,
    "lora_dropout": 0.05,
    "lora_bias": "none",
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
