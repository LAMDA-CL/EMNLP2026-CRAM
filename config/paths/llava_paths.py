"""Central path configuration for LLaVA backbone assets and project outputs."""
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())

BASE_MODEL_PATH = "/root/autodl-tmp/LLaVa"
CLIP_PATH = "/root/autodl-tmp/CLIP"

PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"

CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
RESULT_DIR = f"{PROJECT_ROOT}/results"
LOG_DIR = f"{PROJECT_ROOT}/logs"

DEEPSPEED_CONFIG = f"{PROJECT_ROOT}/config/deepspeed/zero2.json"
