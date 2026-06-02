def __getattr__(name: str):
    if name in ("LlavaLlamaForCausalLM", "LlavaConfig"):
        from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

        return LlavaLlamaForCausalLM if name == "LlavaLlamaForCausalLM" else LlavaConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
