"""Central path configuration for InternVL backbone (edit paths before running)."""
from .common import CHECKPOINT_DIR, DEEPSPEED_CONFIG, LOG_DIR, PROJECT_ROOT, RESULT_DIR

# HuggingFace snapshot roots (weights live under OpenGVLab/<repo-id>/)
BASE_MODEL_PATH = "/data2/mnt2/zhoudw/zh/InternVL-Chat-ViT-6B-Vicuna-7B/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B"
VISION_TOWER_PATH = "/data2/mnt2/zhoudw/zh/InternViT-6B-224px/OpenGVLab/InternViT-6B-224px"
ROUTING_VISION_TOWER_PATH = "/data2/mnt2/zhoudw/zh/CLIP"
CLIP_PATH = ROUTING_VISION_TOWER_PATH

PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"

# If False: MLLM vision weights come from BASE_MODEL_PATH (InternVL-Chat checkpoint) at load time.
# If True: build vision tower from VISION_TOWER_PATH via load_model() (extra ~12GB load).
# VISION_TOWER_PATH is still used for InternViT config (architecture) when False.
LOAD_VISION_TOWER_SEPARATELY = False

# InternVL-Chat merged checkpoint uses 336px InternViT (576 patches + CLS).
# Only applies when LOAD_VISION_TOWER_SEPARATELY is False; separate tower load uses VISION_TOWER_PATH (224).
EMBEDDED_VISION_IMAGE_SIZE = 336

