"""Annotator registry."""
from __future__ import annotations

import threading

from library.annotate.base import Annotator
from library.annotate.sam3 import Sam3Annotator
from library.annotate.yolo import YoloAnnotator

__all__ = ["Annotator", "YoloAnnotator", "Sam3Annotator", "get_annotator", "annotator_statuses"]

_BUILDERS = {"yolo": YoloAnnotator, "sam3": Sam3Annotator}
_lock = threading.Lock()
_cache: dict[str, Annotator] = {}


def get_annotator(name: str, model_ref: str | None = None) -> Annotator:
    """
    Cached per (name, model).  Model weights are expensive to load, and an
    indexing run annotates thousands of images with the same one.
    """
    key = f"{name}:{model_ref or ''}"
    with _lock:
        if key not in _cache:
            builder = _BUILDERS.get(name)
            if builder is None:
                raise ValueError(f"Unknown annotator '{name}'. Available: {', '.join(_BUILDERS)}")
            _cache[key] = builder(model_ref)
        return _cache[key]


def annotator_statuses() -> list[dict]:
    return [get_annotator(name).status().to_dict() for name in _BUILDERS]
