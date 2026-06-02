"""
Defaults for method: clmoe

CL-MoE (Continual Learning Mixture of Experts LoRA):
Input-dependent per-layer soft-routing to independent LoRA expert branches.
Combined with memory replay for continual instruction tuning.

Hyperparameters aligned with the original CL-MoE (MCITlib):
  lora_r=96, lora_alpha=192 (r*2), expert_num=task_num,
  effective batch = per_device * gpus * grad_acc = 16 * 2 * 2 = 64.
"""
from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION


TRAIN_FLAG_OVERRIDES = {
    "--method": "clmoe",
    # lora_r=96 is divisible by common task counts (4, 6, 8)
    # lora_alpha follows the original CL-MoE formula: alpha = r * 2
    "--freeze_mm_mlp_adapter": "True",
    "--mm_projector_lr": "2e-5",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--weight_decay": "0.0",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    "clmethod": "clmoe",
    "batch_size": 1,
}

# per_device_train_batch_size per task (original CL-MoE uses 16)
TRAIN_BATCH_SIZES = {
    "coin": {i: 4 for i in range(8)},
    "ucit": {i: 4 for i in range(6)},
}

METHOD_CONFIG = {
    "cur_task": 0,
    "lora_r": 96,
    "lora_alpha": 192,
    "lora_dropout": 0.05,
    "task_embedding_dim": 64,
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