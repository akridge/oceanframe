"""
Resolve a model reference to a local file.

Domain models matter more than convenience here.  A COCO-pretrained checkpoint
is worse than useless on reef imagery — it labels fish "airplane" and coral
"skateboard" with high confidence — so the fix is to point the annotator at a
model trained on this domain.  NOAA publishes one in the same open bucket as
the imagery::

    LIB_YOLO_MODEL=gs://nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/models/yolo11-esa-icra-detector.pt

so this module accepts a ``gs://`` reference and caches the weights on disk.
"""
from __future__ import annotations

import threading
from pathlib import Path

from library import settings

_lock = threading.Lock()


def resolve(ref: str) -> str:
    """
    Turn a model reference into something Ultralytics can open.

    * ``gs://bucket/key``      → downloaded into the model cache, path returned
    * an existing local path   → returned unchanged
    * a bare name (yolo11n.pt) → returned unchanged, so Ultralytics can fetch it
    """
    ref = (ref or "").strip()
    if not ref.startswith("gs://"):
        return ref

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = settings.MODEL_DIR / Path(ref).name

    with _lock:
        if target.exists() and target.stat().st_size > 0:
            return str(target)

        from library.storage import get_backend  # noqa: PLC0415

        bucket_root = "gs://" + ref[5:].split("/", 1)[0]
        data = get_backend(bucket_root).read_bytes(ref)
        # Write via a temp name so an interrupted download is never mistaken
        # for a cached model on the next run.
        staging = target.with_suffix(target.suffix + ".partial")
        staging.write_bytes(data)
        staging.replace(target)
        return str(target)


def cached_models() -> list[dict]:
    """What is already on disk, for the UI's model panel."""
    if not settings.MODEL_DIR.exists():
        return []
    return [
        {"name": p.name, "path": str(p), "bytes": p.stat().st_size}
        for p in sorted(settings.MODEL_DIR.glob("*.pt"))
    ]
