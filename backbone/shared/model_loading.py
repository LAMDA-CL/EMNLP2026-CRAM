import torch
import transformers
from typing import Dict, Optional

from backbone.shared.multimodal.data_processor import smart_tokenizer_and_embedding_resize
from backbone.shared.multimodal import conversation as conversation_lib
from config.backbone.registry import (
    get_load_vision_tower_separately,
    import_mllm_model_class,
    resolve_backbone_id,
)


def setup_quantization(training_args, compute_dtype) -> Dict:
    bnb_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))
    return bnb_args


def _is_multimodal_checkpoint(path: str, backbone_id: Optional[str] = None) -> bool:
    p = (path or "").lower()
    bid = resolve_backbone_id(backbone_id)
    if bid == "internvl":
        return "intern" in p or "llava" in p
    return "llava" in p


def _apply_local_tower_paths_to_config(
    config,
    *,
    vision_tower: Optional[str] = None,
    routing_vision_tower: Optional[str] = None,
    text_tower: Optional[str] = None,
) -> None:
    """Override Hub ids in a saved config with local CLI paths (before model __init__)."""
    if vision_tower:
        setattr(config, "mm_vision_tower", vision_tower)
    if routing_vision_tower:
        setattr(config, "routing_vision_tower", routing_vision_tower)
    if text_tower:
        setattr(config, "mm_text_tower", text_tower)
        if not hasattr(config, "mm_text_select_layer"):
            setattr(config, "mm_text_select_layer", -1)


def load_pretrained_model(
    model_name_or_path,
    training_args,
    bnb_args,
    has_vision=False,
    backbone_id: Optional[str] = None,
    vision_tower: Optional[str] = None,
    routing_vision_tower: Optional[str] = None,
    text_tower: Optional[str] = None,
):
    ModelClass = import_mllm_model_class(backbone_id)
    if has_vision and _is_multimodal_checkpoint(model_name_or_path, backbone_id):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        _apply_local_tower_paths_to_config(
            config,
            vision_tower=vision_tower,
            routing_vision_tower=routing_vision_tower,
            text_tower=text_tower,
        )
        if resolve_backbone_id(backbone_id) == "internvl":
            from config.backbone.registry import get_embedded_vision_image_size

            load_sep = get_load_vision_tower_separately(backbone_id)
            setattr(config, "load_vision_tower_separately", load_sep)
            if not load_sep:
                setattr(
                    config,
                    "embedded_vision_image_size",
                    get_embedded_vision_image_size(backbone_id),
                )
        model = ModelClass.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_args,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_args
        )
    model.config.use_cache = False
    return model


def load_tokenizer(model_name_or_path, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    return tokenizer


def setup_tokenizer(tokenizer, model, version):
    if version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


def load_clip_tokenizer(text_tower_path, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        text_tower_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    return tokenizer


def initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer):
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.get_model().initialize_text_modules(model_args=model_args, fsdp=training_args.fsdp)
    text_tower = model.get_text_tower()
    if text_tower is not None:
        text_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.get_model().initialize_routing_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    routing_tower = model.get_model().get_routing_vision_tower()
    if routing_tower is not None:
        routing_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    return model, data_args


def apply_mm_projector_trainability(model, training_args) -> None:
    """
    Apply ``mm_projector`` ``requires_grad`` after CL integration may freeze the full model.

    - ``tune_mm_mlp_adapter``: train projector only (LLaVA projector-tuning mode).
    - ``freeze_mm_mlp_adapter``: freeze projector (default for most CL methods via ``run.py``).
    - otherwise: projector parameters are trainable (use with ``--mm_projector_lr``).
    """
    inner = getattr(model, "_base_model", model)
    get_model = getattr(inner, "get_model", None)
    if get_model is None:
        return
    meta = get_model()
    projector = getattr(meta, "mm_projector", None)
    if projector is None:
        return

    tune_only = bool(getattr(training_args, "tune_mm_mlp_adapter", False))
    freeze = bool(getattr(training_args, "freeze_mm_mlp_adapter", False))

    if tune_only:
        for _, p in model.named_parameters():
            p.requires_grad = False
        for p in projector.parameters():
            p.requires_grad = True
    elif freeze:
        for p in projector.parameters():
            p.requires_grad = False
    else:
        for p in projector.parameters():
            p.requires_grad = True


def adjust_precision(model, training_args):
    """Align LoRA adapter dtypes with bf16/fp16 training (8-bit/4-bit LLM + QLoRA)."""
    if training_args.bits not in (4, 8):
        return model
    from PEFT.tuners.standard.lora import LoraLayer

    adapter_dtype = (
        torch.bfloat16
        if training_args.bf16
        else torch.float16
        if training_args.fp16
        else None
    )
    for name, module in model.named_modules():
        if adapter_dtype is not None and isinstance(module, LoraLayer):
            module.to(adapter_dtype)
        if "norm" in name:
            module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight") and adapter_dtype is not None:
                if module.weight.dtype == torch.float32:
                    module.to(adapter_dtype)
    return model


def prepare_model_for_kbit(model, training_args, compute_dtype):
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    return model


def setup_gradient_checkpointing(model, training_args):
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model
