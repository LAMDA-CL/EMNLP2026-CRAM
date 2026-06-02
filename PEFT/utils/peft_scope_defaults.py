"""
Shared PEFT path filters for ``METHOD_CONFIG``: matches PEFT’s built-in LLaVA defaults so definitions stay in sync.
"""

from __future__ import annotations

from typing import Final, List

from PEFT.utils.llava_peft_scope import DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS

EXCLUDE_FOR_LLM_ONLY_INJECTION: Final[List[str]] = list(DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS)
