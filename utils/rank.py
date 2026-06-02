"""Distributed helpers: treat rank as local main when ``LOCAL_RANK`` is unset, -1, or 0."""
from __future__ import annotations

import os


def is_local_main_process() -> bool:
    v = os.environ.get("LOCAL_RANK", "-1")
    try:
        return int(v) in (-1, 0)
    except ValueError:
        return True
