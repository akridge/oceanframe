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

import logging
import tempfile
import threading
from pathlib import Path

from library import settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_warned = False


def _writable_cache_dir() -> Path:
    """
    Where downloaded weights are written.

    ``LIB_MODEL_DIR`` is frequently a read-only or differently-owned mount — a
    bind-mounted ./models owned by the host user while the container runs as
    another uid is the common case — so fall back to a temp dir rather than
    failing an annotation run. Weights already present in the configured
    directory are still used from there; only downloads move.
    """
    global _warned
    primary = settings.MODEL_DIR
    try:
        primary.mkdir(parents=True, exist_ok=True)
        probe = primary / ".oceanframe-write-test"
        probe.touch()
        probe.unlink()
        return primary
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "oceanframe-models"
        fallback.mkdir(parents=True, exist_ok=True)
        if not _warned:
            _warned = True
            log.warning(
                "%s is not writable (%s); caching downloaded weights in %s instead. "
                "They will be re-downloaded when the container restarts — bind-mount a "
                "writable directory, or build with --build-arg UID=$(id -u), to keep them.",
                primary, exc.strerror or exc, fallback,
            )
        return fallback


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

    name = Path(ref).name

    with _lock:
        # Prefer weights already supplied in the configured directory, even when
        # it is read-only — that is how you ship sam3.pt into a container.
        supplied = settings.MODEL_DIR / name
        if supplied.exists() and supplied.stat().st_size > 0:
            return str(supplied)

        target = _writable_cache_dir() / name
        if target.exists() and target.stat().st_size > 0:
            return str(target)

        from library.storage import get_backend  # noqa: PLC0415

        bucket_root = "gs://" + ref[5:].split("/", 1)[0]
        data = get_backend(bucket_root).read_bytes(ref)
        # Write via a temp name so an interrupted download is never mistaken
        # for a cached model on the next run.
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".partial")
        staging.write_bytes(data)
        staging.replace(target)
        return str(target)


def cached_models() -> list[dict]:
    """What is already on disk, for the UI's model panel."""
    seen: dict[str, dict] = {}
    for directory in (settings.MODEL_DIR, Path(tempfile.gettempdir()) / "oceanframe-models"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.pt")):
            seen.setdefault(path.name, {
                "name": path.name, "path": str(path), "bytes": path.stat().st_size,
            })
    return list(seen.values())
