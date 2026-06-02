"""InternVL backbone package (lazy exports)."""


def __getattr__(name: str):
    if name == "LlavaLlamaForCausalLM":
        from .model import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
