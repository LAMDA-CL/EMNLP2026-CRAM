"""LLaVA backbone constants (aligned with Vicuna-7B / 32-layer Llama)."""

BACKBONE_ID = "llava"
DEFAULT_CONV_MODE = "vicuna_v1"
CHECKPOINT_SUFFIX = "_llava"

# CLIP routing feature dim (text tower + routing vision tower)
ROUTING_FEATURE_DIM = 768
CLIP_FEATURE_DIM = ROUTING_FEATURE_DIM

NUM_HIDDEN_LAYERS = 32
LAST_LORA_BLOCK_INDEX = NUM_HIDDEN_LAYERS - 1

TRAIN_BACKBONE_FLAGS = {
    "version": "v1",
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_layer": "-2",
    "mm_use_im_start_end": "False",
    "mm_use_im_patch_token": "False",
    "image_aspect_ratio": "pad",
    "learning_rate": "2e-4",
}
