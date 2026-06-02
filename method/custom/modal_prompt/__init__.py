"""
ModalPrompt method package.

Keep lightweight, avoid import-time side effects (e.g. early PEFT/torch imports).
Real registration happens on demand in `method/custom/modal_prompt/integration.py`.
"""
