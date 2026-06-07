"""Shared project paths (checkpoints, results, logs). Benchmark data paths live in config/benchmarks/."""
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())

CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
RESULT_DIR = f"{PROJECT_ROOT}/results"
LOG_DIR = f"{PROJECT_ROOT}/logs"

DEEPSPEED_CONFIG = f"{PROJECT_ROOT}/config/deepspeed/zero2.json"
