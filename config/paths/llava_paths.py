"""Central path configuration for LLaVA backbone (edit paths before running)."""
from .common import CHECKPOINT_DIR, DEEPSPEED_CONFIG, LOG_DIR, PROJECT_ROOT, RESULT_DIR

BASE_MODEL_PATH = "/data2/mnt2/zhoudw/zh/LLaVa"
CLIP_PATH = "/data2/mnt2/zhoudw/zh/CLIP"
VISION_TOWER_PATH = CLIP_PATH
ROUTING_VISION_TOWER_PATH = CLIP_PATH

PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"
