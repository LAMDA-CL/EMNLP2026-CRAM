from .config_loader import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    load_config,
    merge_benchmark_task_num_into,
    merge_method_config_into,
)

__all__ = [
    "ModelArguments",
    "DataArguments",
    "TrainingArguments",
    "load_config",
    "merge_benchmark_task_num_into",
    "merge_method_config_into",
    "load_model_for_train",
    "load_model_for_inference",
    "load_from_checkpoint",
    "save_model",
    "make_supervised_data_module",
    "LengthGroupedSampler",
    "smart_tokenizer_and_embedding_resize",
]


def __getattr__(name: str):
    """Lazy-import heavy modules to avoid import-time TF/torch.compile/sentencepiece conflicts."""
    if name == "load_model_for_train":
        from .load_model import load_model_for_train

        return load_model_for_train
    if name == "load_model_for_inference":
        from .load_model import load_model_for_inference

        return load_model_for_inference
    if name == "load_from_checkpoint":
        from .load_model import load_from_checkpoint

        return load_from_checkpoint
    if name == "save_model":
        from .save_checkpoint import save_model

        return save_model
    if name == "make_supervised_data_module":
        from backbone.shared.data import make_supervised_data_module

        return make_supervised_data_module
    if name == "LengthGroupedSampler":
        from backbone.shared.data import LengthGroupedSampler

        return LengthGroupedSampler
    if name == "smart_tokenizer_and_embedding_resize":
        from backbone.shared.multimodal.data_processor import (
            smart_tokenizer_and_embedding_resize,
        )

        return smart_tokenizer_and_embedding_resize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
