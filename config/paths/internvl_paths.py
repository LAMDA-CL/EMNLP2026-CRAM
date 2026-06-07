"""Central path configuration for InternVL backbone (edit paths before running)."""
from .common import CHECKPOINT_DIR, DEEPSPEED_CONFIG, LOG_DIR, PROJECT_ROOT, RESULT_DIR

# HuggingFace snapshot roots (weights live under OpenGVLab/<repo-id>/)
BASE_MODEL_PATH = "/root/autodl-tmp/InternVL-Chat-ViT-6B-Vicuna-7B"
VISION_TOWER_PATH = "/root/autodl-tmp/InternViT-6B-224px"
ROUTING_VISION_TOWER_PATH = "/root/autodl-tmp/CLIP"
CLIP_PATH = ROUTING_VISION_TOWER_PATH

PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"

# If False: MLLM vision weights come from BASE_MODEL_PATH (InternVL-Chat checkpoint) at load time.
# If True: build vision tower from VISION_TOWER_PATH via load_model() (extra ~12GB load).
# VISION_TOWER_PATH is still used for InternViT config (architecture) when False.
#
# Zeroshot / eval baseline: use False — InternVL-Chat mm_projector expects 336px / 576 patches.
# Separate 224px InternViT (256 patches) causes severe VQA degradation (answers like "." / "Ъ").
# CL training may set True when the method intentionally swaps in a separate ViT.
LOAD_VISION_TOWER_SEPARATELY = False

# InternVL-Chat merged checkpoint uses 336px InternViT (576 patches + CLS).
# Applies only when LOAD_VISION_TOWER_SEPARATELY is False (ViT weights from BASE_MODEL_PATH).
# When True, infer/train load ViT from VISION_TOWER_PATH (e.g. 224px InternViT-6B-224px).
EMBEDDED_VISION_IMAGE_SIZE = 336

