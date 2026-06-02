"""
Central place to edit run defaults.

Rule:
- If a parameter is present here, it becomes the default.
- If you pass the same parameter via CLI, CLI overrides it.
"""

TRAIN_DEFAULTS = {
    "benchmark": "ucit",
    "gpus": "0,1",
    "port": 29602,
    "backbone": "internvl",
    # True → training subprocess gets PYPRISM_LOG_LEVEL=DEBUG (see run.py / train.py). Does not change batch size.
    "debug": False,
    # hide_llava, olora, replay_lora, ...
    "method": "replay_lora",
    # UCIT: when True, append _sub to train/test/eval *.json paths (canonical names in config/benchmarks/UCIT.py).
    "use_sub_dataset": True,
}

# Extra args appended at the end of the training command.
# Example: ["--save_steps", "100", "--evaluation_strategy", "steps"]
TRAIN_EXTRA_ARGS: list[str] = []

# Training compute precision (passed to train_mem.py via run.py).
#   "bf16" — full weights + DeepSpeed/HF bf16 mixed precision (bits=16, highest VRAM)
#   "fp16" — full weights + fp16 mixed precision
#   "8bit" — bitsandbytes 8-bit LLM (QLoRA-style) + bf16 compute (default, saves VRAM)
#   "4bit" — bitsandbytes 4-bit LLM + bf16 compute
TRAIN_PRECISION = "8bit"

# ===== Argument defaults (infer) =====
# Inference compute precision for load_model_for_inference (LLM + multimodal towers).
#   "bf16" — bfloat16 weights (~2x VRAM vs 4bit; recommended default)
#   "fp16" — float16
#   "4bit" / "8bit" — bitsandbytes quantized LLM only (vision towers stay fp16/bf16)
# InternVL zeroshot/VQA: avoid "4bit" (often ~0% accuracy); use "bf16" or "8bit".
INFER_PRECISION = "8bit"

INFER_DEFAULTS = {
    "benchmark": "ucit",
    "gpus": "0,1",
    "backbone": "internvl",
    "checkpoint_task": "0",
    "checkpoint_suffix": "_internvl",
    "stage": "last",
    "method": "replay_lora",
    "temperature": "0",
    "use_sub_dataset": True,
}






# ===== Argument defaults (train) =====
# TRAIN_DEFAULTS = {
#     "benchmark": "ucit",
#     "gpus": "0,1",
#     "port": 29602,
#     "backbone": "llava",
#     # True → training subprocess gets PYPRISM_LOG_LEVEL=DEBUG (see run.py / train.py). Does not change batch size.
#     "debug": False,
#     # hide_llava, olora, replay_lora, ...
#     "method": "same",
#     # UCIT: when True, append _sub to train/test/eval *.json paths (canonical names in config/benchmarks/UCIT.py).
#     "use_sub_dataset": True,
# }

# # Extra args appended at the end of the training command.
# # Example: ["--save_steps", "100", "--evaluation_strategy", "steps"]
# TRAIN_EXTRA_ARGS: list[str] = []

# # ===== Argument defaults (infer) =====
# INFER_DEFAULTS = {
#     "benchmark": "ucit",
#     "gpus": "0,1",
#     "backbone": "llava",
#     "checkpoint_task": "5",
#     "checkpoint_suffix": "_llava",
#     "stage": "last",
#     "method": "same",
#     "temperature": "0",
#     "use_sub_dataset": True,
# }


BACKBONE_DEFAULT = "llava"

# Notes:
# - Train precision: `TRAIN_PRECISION` (bf16 / fp16 / 8bit / 4bit); override with `run.py train --train-precision`.
# - Infer precision: `INFER_PRECISION`; override with `run.py infer --load-8bit` / `--load-4bit`.
# - Train batch sizes are method-specific: `config/methods/<method>.py` → `TRAIN_BATCH_SIZES`.
# - Gradient accumulation: `TRAIN_GRADIENT_ACCUMULATION_STEPS` (int or per-benchmark/task dict).
#   Use `-1` to auto-set `global_batch // (per_device_batch × num_gpus)`; target global batch from
#   `TRAIN_GLOBAL_BATCH_SIZE` or benchmark task `batch_size` (default grad_acc=1 if unset).
# - Default backbone: BACKBONE_DEFAULT or `--backbone` / PRISM_BACKBONE env.
# - Default conversation template: active backbone config → DEFAULT_CONV_MODE; override at infer with `--conv-mode`.
# - `use_sub_dataset`: for **ucit** only, runtime add/remove `_sub` on data paths; see `utils/sub_dataset.py`.
