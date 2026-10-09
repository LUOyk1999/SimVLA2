"""Dataset-handler registry.

``get_handler_cls(dataset_name)`` maps a meta's ``dataset_name`` (lowercase
``<bench>``) to the handler class that knows how to iterate that dataset.
A new handler is added to ``_REGISTRY`` below, so the trainer stays generic.
"""
from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler
from .calvin_hdf5 import CalvinHandler
from .libero_hdf5 import LiberoHDF5Handler
from .bridge_rlds import BridgeV2Handler
from .fractal_rlds import FractalHandler

# Registry for dataset handlers. Keys match `dataset_name` in a meta file
# (lowercase <bench>).
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    "bridge":   BridgeV2Handler,
    "libero":   LiberoHDF5Handler,
    "calvin":   CalvinHandler,
    "fractal":  FractalHandler,
}


def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domain_handler/registry.py."
        )
