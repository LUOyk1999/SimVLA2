"""Domain handlers for different dataset formats.

:func:`get_handler_cls` (see ``registry.py``) maps a meta's ``dataset_name``
to its handler class. Every handler module is imported up front; TensorFlow,
which only the RLDS handlers need, is imported on first use instead.
"""
from .base import DomainHandler
from .registry import get_handler_cls

__all__ = [
    "DomainHandler",
    "get_handler_cls",
]
