"""
Runtime registration for custom PEFT configs/models so new methods stay under ``method/<name>/``.
"""
from typing import Optional, Type


def register_peft_extension(
    peft_type: str,
    config_cls: Type,
    tuner_model_cls: Optional[Type] = None,
    task_type: Optional[str] = None,
    task_peft_model_cls: Optional[Type] = None,
) -> None:
    """
    Inject custom PEFT components into PEFT internal maps (idempotent).

    Args:
        peft_type: Custom ``peft_type`` string (e.g. ``MOE_LORA_HiDe``).
        config_cls: Subclass of ``PeftConfig``.
        tuner_model_cls: Tuner module class used inside ``PeftModel``.
        task_type: Optional task-type key (e.g. ``CAUSAL_LM_HiDe``).
        task_peft_model_cls: Optional ``PeftModel`` subclass for ``task_type``.
    """
    from PEFT import mapping as peft_mapping
    from PEFT import peft_model as peft_model_module

    peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls

    if tuner_model_cls is not None:
        peft_model_module.PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = tuner_model_cls

    if task_type is not None and task_peft_model_cls is not None:
        peft_mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type] = task_peft_model_cls
