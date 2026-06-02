"""
Defaults for method: zeroshot — plain LLaVA baseline (inference / eval only).

Not used for CL training: do not ``run.py train --method zeroshot``.
``TRAIN_FLAG_OVERRIDES`` / ``TRAIN_BATCH_SIZES`` are intentionally empty; only ``INFER_DEFAULTS`` matters for ``run.py infer``.
"""

TRAIN_FLAG_OVERRIDES: dict[str, str] = {}
TRAIN_EXTRA_ARGS: list[str] = []
TRAIN_BATCH_SIZES: dict = {}

INFER_DEFAULTS = {
    "batch_size": 3,
}

METHOD_CONFIG: dict = {}
