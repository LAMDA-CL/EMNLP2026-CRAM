"""InternVL-Chat-ViT-6B-Vicuna-7B backbone constants."""

BACKBONE_ID = "internvl"
DEFAULT_CONV_MODE = "vicuna_v1"
CHECKPOINT_SUFFIX = "_internvl"

# CLIP text / routing vision embedding dim (768 for clip-vit-large-patch14-336)
ROUTING_FEATURE_DIM = 768
CLIP_FEATURE_DIM = ROUTING_FEATURE_DIM

NUM_HIDDEN_LAYERS = 32
LAST_LORA_BLOCK_INDEX = NUM_HIDDEN_LAYERS - 1

TRAIN_BACKBONE_FLAGS = {
    "version": "v1",
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_layer": "-4",
    "mm_use_im_start_end": "False",
    "mm_use_im_patch_token": "False",
    "image_aspect_ratio": "pad",
    "learning_rate": "1e-4",
}
