import os


def is_intern_vit_6b_model(vision_tower_name: str) -> bool:
    model_names = ["intern_vit_6b", "internvit_6b", "InternViT-6B", "internvit6b"]
    return any(name in vision_tower_name for name in model_names)


def is_local_pretrained_dir(path: str) -> bool:
    """True if ``path`` is a directory with a HuggingFace-style ``config.json``."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "config.json"))


def pretrained_load_kwargs(path: str) -> dict:
    """Use offline loading when weights are already on disk (avoid Hub lookups)."""
    if is_local_pretrained_dir(path):
        return {"local_files_only": True}
    return {}


def intern_vit_image_size(
    vision_tower_name: str,
    config=None,
    *,
    embedded_in_chat: bool = False,
    embedded_image_size: int | None = None,
) -> int:
    """Resolve InternViT input resolution (224 / 336 / 448) from config or path name."""
    if embedded_in_chat:
        if embedded_image_size is not None:
            return int(embedded_image_size)
        from config.backbone.registry import get_embedded_vision_image_size

        return get_embedded_vision_image_size("internvl")
    if config is not None:
        size = getattr(config, "image_size", None)
        if size is not None:
            return int(size)
    name = vision_tower_name or ""
    if "448" in name:
        return 448
    if "224" in name:
        return 224
    return 336


def build_intern_vit_image_processor(
    vision_tower_name: str,
    config=None,
    *,
    embedded_in_chat: bool = False,
    embedded_image_size: int | None = None,
):
    """CLIP-style preprocessor aligned with the InternViT checkpoint."""
    from transformers import CLIPImageProcessor

    crop_size = intern_vit_image_size(
        vision_tower_name,
        config=config,
        embedded_in_chat=embedded_in_chat,
        embedded_image_size=embedded_image_size,
    )
    if embedded_in_chat:
        return CLIPImageProcessor(
            crop_size=crop_size,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            size=crop_size,
        )
    load_kw = pretrained_load_kwargs(vision_tower_name)
    if is_local_pretrained_dir(vision_tower_name):
        preproc_cfg = os.path.join(vision_tower_name, "preprocessor_config.json")
        if os.path.isfile(preproc_cfg):
            return CLIPImageProcessor.from_pretrained(vision_tower_name, **load_kw)
    return CLIPImageProcessor(
        crop_size=crop_size,
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size=crop_size,
    )
