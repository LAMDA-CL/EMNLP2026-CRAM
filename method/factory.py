# method/factory.py
"""
Factory for continual-learning integrations.

Loads implementations from ``method/custom/*/integration.py`` via registration.
"""
from typing import Any, Dict, Optional, Set
import importlib
from pathlib import Path

from .base.integration import CLIntegration


class CLMethodFactory:
    """
    Builds ``CLIntegration`` instances from method names / aliases.
    """

    # name / alias -> integration class
    _method_registry: Dict[str, type] = {}
    _discovered_modules: Set[str] = set()
    _discovery_done: bool = False

    @classmethod
    def register(cls, *method_names: str):
        """
        Decorator to register a CL integration.

        Usage: ``@CLMethodFactory.register("sp", "alias")``
        """
        def decorator(integration_class: type):
            if not method_names:
                raise ValueError("register() requires at least one method_name")
            for method_name in method_names:
                cls._method_registry[method_name.lower()] = integration_class
            return integration_class
        return decorator
