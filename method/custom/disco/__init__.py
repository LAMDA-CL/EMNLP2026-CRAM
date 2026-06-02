"""
DISCO method package.

Keep lightweight, avoid import-time side effects (e.g. early PEFT/torch imports).
Real registration happens on demand in `method/custom/disco/integration.py`.
"""
