"""
Ingestion pipeline: bucket -> catalog.

    list -> (fetch, decode, measure, thumbnail) x N threads -> embed -> upsert

Downloads dominate the wall clock on a GCS source, so fetch/decode/measure runs
on a thread pool while embedding and the SQLite writes stay on the job thread —
one writer, no lock contention, and the embedder sees whole batches instead of
single images.

Re-running is cheap: an object whose ``etag`` (GCS generation) is unchanged is
skipped before a single byte is downloaded.
"""
from __future__ import annotations

import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Iterator

from PIL import Image, ImageOps

from library import db, settings, tags
from library.embed import get_embedder
from library.jobs import Job
from library.storage import StorageBackend, backend_kind, get_backend, normalise_root
from library.models import ObjectRef
from library.quality import analyse
from library.vectorstore import get_store

# A 25000x25000 JPEG is a decompression bomb, not a survey photo.
Image.MAX_IMAGE_PIXELS = 300_000_000


@dataclass
class Prepared:
    """One object after download + decode, ready to embed and write."""
    ref:     ObjectRef
    image:   Image.Image | None
    metrics: object | None
    thumb:   str
    error:   str = ""


# ── Source registration ───────────────────────────────────────────────────────

def ensure_source(root: str, label: str = "") -> tuple[int, str]:
    """Upsert a source row.  Returns (source_id, normalised_root)."""
    root = normalise_root(root)
    with db.write() as conn:
        row = conn.execute("SELECT id FROM sources WHERE root = ?", (root,)).fetchone()
        if row:
            return int(row["id"]), root
        cur = conn.execute(
            "INSERT INTO sources(kind, root, label, added_at) VALUES (?, ?, ?, ?)",
            (backend_kind(root), root, label or PurePosixPath(root).name or root, time.time()),
        )
        return int(cur.lastrowid), root


def list_sources() -> list[dict]:
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT s.*, COUNT(a.id) AS n,
               SUM(CASE WHEN a.status='ok' THEN 1 ELSE 0 END) AS n_ok
          FROM sources s LEFT JOIN assets a ON a.source_id = s.id
         GROUP BY s.id ORDER BY s.added_at
        """
    ).fetchall()
    return [
        {
            "id": r["id"], "kind": r["kind"], "root": r["root"], "label": r["label"],
            "assets": r["n"] or 0, "indexed": r["n_ok"] or 0, "scanned_at": r["scanned_at"],
        }
        for r in rows
    ]


# ── Thumbnails ────────────────────────────────────────────────────────────────

def _thumb_path_for(uri: str) -> tuple[str, str]:
    """(absolute path, catalog-relative path).  Sharded so no directory blows up."""
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()
    rel = f"{digest[:2]}/{digest}.jpg"
    return str(settings.THUMB_DIR / rel), rel


def thumb_abs_path(rel: str) -> str:
    return str(settings.THUMB_DIR / rel)


def _write_thumb(image: Image.Image, uri: str) -> str:
    abs_path, rel = _thumb_path_for(uri)
    thumb = image.copy()
    thumb.thumbnail((settings.THUMB_WIDTH, settings.THUMB_WIDTH * 4), Image.LANCZOS)
    target = settings.THUMB_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    thumb.convert("RGB").save(abs_path, format="JPEG", quality=settings.THUMB_QUALITY, optimize=True)
    return rel


# ── Fetch + measure ───────────────────────────────────────────────────────────

def _downscale(image: Image.Image) -> Image.Image:
    """Bound the working image so an 8000px drone frame does not blow up memory."""
    image = ImageOps.exif_transpose(image)
    longest = max(image.size)
    if longest > settings.WORK_MAX_EDGE:
        scale = settings.WORK_MAX_EDGE / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS
        )
    return image


def _prepare(backend: StorageBackend, ref: ObjectRef) -> Prepared:
    try:
        raw = backend.read_bytes(ref.uri)
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            full_size = opened.size
            image = _downscale(opened).convert("RGB")
        metrics = analyse(image)
        # Report the *source* dimensions, not the working copy's.
        metrics.width, metrics.height = full_size
        return Prepared(ref=ref, image=image, metrics=metrics, thumb=_write_thumb(image, ref.uri))
    except Exception as exc:
        return Prepared(ref=ref, image=None, metrics=None, thumb="", error=f"{type(exc).__name__}: {exc}")


# ── Writing ───────────────────────────────────────────────────────────────────

def _upsert(conn, source_id: int, item: Prepared, embed_row: int, embed_model: str) -> int:
    ref = item.ref
    folder = str(PurePosixPath(ref.key).parent)
    folder = "" if folder == "." else folder
    name = PurePosixPath(ref.key).name
    ext = PurePosixPath(name).suffix.lower()
    now = time.time()

    if item.error or item.metrics is None:
        conn.execute(
            """
            INSERT INTO assets(source_id, uri, folder, name, ext, size, etag, mtime, status, error, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'error', ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
                etag=excluded.etag, mtime=excluded.mtime, size=excluded.size,
                status='error', error=excluded.error, indexed_at=excluded.indexed_at
            """,
            (source_id, ref.uri, folder, name, ext, ref.size, ref.etag, ref.mtime, item.error[:500], now),
        )
    else:
        m = item.metrics
        conn.execute(
            """
            INSERT INTO assets(source_id, uri, folder, name, ext, size, etag, mtime,
                               width, height, phash, blur, brightness, contrast, color_cast,
                               quality, thumb_path, embed_row, embed_model, indexed_at, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', '')
            ON CONFLICT(uri) DO UPDATE SET
                etag=excluded.etag, mtime=excluded.mtime, size=excluded.size,
                width=excluded.width, height=excluded.height, phash=excluded.phash,
                blur=excluded.blur, brightness=excluded.brightness, contrast=excluded.contrast,
                color_cast=excluded.color_cast, quality=excluded.quality,
                thumb_path=excluded.thumb_path, embed_row=excluded.embed_row,
                embed_model=excluded.embed_model, indexed_at=excluded.indexed_at,
                status='ok', error=''
            """,
            (source_id, ref.uri, folder, name, ext, ref.size, ref.etag, ref.mtime,
             m.width, m.height, m.phash, m.blur, m.brightness, m.contrast, m.color_cast,
             m.quality, item.thumb, embed_row, embed_model, now),
        )

    asset_id = int(conn.execute("SELECT id FROM assets WHERE uri = ?", (ref.uri,)).fetchone()["id"])
    derived = tags.path_tags(ref.key)
    if derived:
        tags.add_tags(conn, [asset_id], derived, kind="path", origin="path")
    else:
        db.fts_refresh(conn, [asset_id])
    return asset_id


# ── Indexing ──────────────────────────────────────────────────────────────────

def _pending_refs(
    conn, source_id: int, backend: StorageBackend, force: bool, limit: int, job: Job
) -> tuple[list[ObjectRef], set[str], int]:
    """Split the listing into work and skips.  Also returns every URI seen."""
    known = {
        r["uri"]: (r["etag"], r["status"])
        for r in conn.execute(
            "SELECT uri, etag, status FROM assets WHERE source_id = ?", (source_id,)
        )
    }
    pending: list[ObjectRef] = []
    seen: set[str] = set()
    skipped = 0

    for ref in backend.list_objects(settings.IMAGE_EXTENSIONS):
        if job.cancelled:
            break
        seen.add(ref.uri)
        prior = known.get(ref.uri)
        if not force and prior and prior[0] == ref.etag and prior[1] == "ok":
            skipped += 1
        else:
            pending.append(ref)
        if limit and len(pending) >= limit:
            break
        if (len(seen) % 2000) == 0:
            job.log(f"Listing… {len(seen):,} objects ({len(pending):,} to index)")
    return pending, seen, skipped


def _batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def index_source(
    job: Job,
    root: str,
    *,
    force: bool = False,
    limit: int = 0,
    label: str = "",
    prune: bool = True,
) -> dict:
    """Crawl a source and bring the catalog up to date with it."""
    db.init_db()
    settings.ensure_dirs()

    root = normalise_root(root)
    backend = get_backend(root)
    if not backend.exists():
        raise RuntimeError(
            f"Cannot read {backend.describe()}. "
            + ("Check the bucket name and that Application Default Credentials are set "
               "(gcloud auth application-default login), and that the account has "
               "roles/storage.objectViewer." if backend.kind == "gcs"
               else "Check the path exists and is a directory.")
        )

    source_id, root = ensure_source(root, label)
    embedder = get_embedder()
    store = get_store()
    wiped = store.ensure_model(embedder.dim, embedder.name)
    if wiped:
        job.log(f"Embedding model changed to {embedder.describe()} — clearing old vectors")
        with db.write() as conn:
            conn.execute("UPDATE assets SET embed_row = -1, embed_model = ''")

    job.log(f"Listing {backend.describe()}…")
    conn = db.connect()
    max_assets = limit or settings.MAX_ASSETS
    pending, seen, skipped = _pending_refs(conn, source_id, backend, force, max_assets, job)

    job.set_total(len(pending))
    job.log(f"{len(pending):,} to index, {skipped:,} unchanged")

    indexed = failed = 0
    workers = max(1, settings.CRAWL_WORKERS)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch") as pool:
        for batch in _batched(pending, settings.INDEX_BATCH):
            if job.cancelled:
                break
            prepared = list(pool.map(lambda ref: _prepare(backend, ref), batch))

            good = [p for p in prepared if p.error == "" and p.image is not None]
            rows: dict[str, int] = {}
            if good:
                try:
                    vectors = embedder.embed_images([p.image for p in good])
                    assigned = store.add(vectors)
                    rows = {p.ref.uri: r for p, r in zip(good, assigned)}
                except Exception as exc:
                    job.log(f"Embedding failed for a batch, continuing without vectors: {exc}")

            with db.write() as write_conn:
                for item in prepared:
                    _upsert(write_conn, source_id, item, rows.get(item.ref.uri, -1), embedder.name)
                    if item.error:
                        failed += 1
                    else:
                        indexed += 1

            for item in prepared:
                item.image = None            # release the decoded batch promptly
            job.advance(len(prepared), f"{indexed:,} indexed · {failed:,} failed")

    missing = 0
    if prune and not job.cancelled and not max_assets:
        missing = _mark_missing(source_id, seen)

    with db.write() as write_conn:
        write_conn.execute("UPDATE sources SET scanned_at = ? WHERE id = ?", (time.time(), source_id))

    return {
        "source_id": source_id, "root": root, "indexed": indexed, "failed": failed,
        "skipped": skipped, "missing": missing, "embedder": embedder.describe(),
        "text_search": embedder.supports_text,
    }


def _mark_missing(source_id: int, seen: set[str]) -> int:
    """Flag catalog rows whose object is gone from the bucket.  Never deletes."""
    conn = db.connect()
    stale = [
        r["id"] for r in conn.execute(
            "SELECT id, uri FROM assets WHERE source_id = ? AND status != 'missing'", (source_id,)
        ) if r["uri"] not in seen
    ]
    if stale:
        with db.write() as write_conn:
            write_conn.executemany(
                "UPDATE assets SET status='missing' WHERE id = ?", [(i,) for i in stale]
            )
    return len(stale)


# ── Re-embedding ──────────────────────────────────────────────────────────────

def embed_missing(job: Job, *, rebuild: bool = False) -> dict:
    """
    Fill in (or rebuild) vectors without re-crawling.

    This is the upgrade path from the hash descriptor to CLIP: install torch,
    set ``LIB_EMBED_BACKEND=clip``, run this, and every thumbnail, quality
    metric and tag is preserved.
    """
    db.init_db()
    embedder = get_embedder()
    store = get_store()
    conn = db.connect()

    if rebuild:
        store.reset(embedder.dim, embedder.name)
        with db.write() as write_conn:
            write_conn.execute("UPDATE assets SET embed_row = -1, embed_model = ''")
    else:
        if store.ensure_model(embedder.dim, embedder.name):
            with db.write() as write_conn:
                write_conn.execute("UPDATE assets SET embed_row = -1, embed_model = ''")

    rows = conn.execute(
        "SELECT id, uri, thumb_path FROM assets WHERE status='ok' AND embed_row < 0 ORDER BY id"
    ).fetchall()
    job.set_total(len(rows))
    job.log(f"Embedding {len(rows):,} assets with {embedder.describe()}")

    done = 0
    for batch in _batched(list(rows), settings.EMBED_BATCH):
        if job.cancelled:
            break
        images, ids = [], []
        for row in batch:
            # Thumbnails are the cheap path: local, already decoded once, and at
            # 320px they are larger than every CLIP input resolution.
            path = thumb_abs_path(row["thumb_path"]) if row["thumb_path"] else ""
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
                ids.append(row["id"])
            except Exception:
                continue
        if not images:
            job.advance(len(batch))
            continue
        vectors = embedder.embed_images(images)
        assigned = store.add(vectors)
        with db.write() as write_conn:
            write_conn.executemany(
                "UPDATE assets SET embed_row = ?, embed_model = ? WHERE id = ?",
                [(r, embedder.name, i) for r, i in zip(assigned, ids)],
            )
        done += len(ids)
        job.advance(len(batch), f"{done:,} embedded")

    return {"embedded": done, "embedder": embedder.describe(), "text_search": embedder.supports_text}


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_assets(
    job: Job,
    asset_ids: Iterable[int],
    *,
    annotator: str = "yolo",
    prompts: list[str] | None = None,
    model_ref: str | None = None,
    replace: bool = True,
) -> dict:
    """Run a model over the given assets and store detections + auto-tags."""
    from library.annotate import get_annotator  # noqa: PLC0415 - keeps torch out of import time

    db.init_db()
    engine = get_annotator(annotator, model_ref)
    state = engine.status()
    if not state.available:
        raise RuntimeError(state.detail)

    ids = list(asset_ids)
    conn = db.connect()
    job.set_total(len(ids))
    job.log(f"Annotating {len(ids):,} assets with {annotator}")

    kind = "concept" if annotator == "sam3" else "class"
    prefix = "concept" if annotator == "sam3" else "class"
    total_dets = 0
    processed = 0

    for batch_ids in _batched(ids, max(1, settings.EMBED_BATCH)):
        if job.cancelled:
            break
        marks = ",".join("?" * len(batch_ids))
        rows = conn.execute(
            f"SELECT id, uri, thumb_path FROM assets WHERE id IN ({marks})", batch_ids
        ).fetchall()

        images, keep_ids = [], []
        for row in rows:
            image = _load_for_model(row)
            if image is not None:
                images.append(image)
                keep_ids.append(row["id"])
        if not images:
            job.advance(len(batch_ids))
            continue

        results = engine.annotate(images, prompts)
        with db.write() as write_conn:
            if replace:
                write_conn.executemany(
                    "DELETE FROM detections WHERE asset_id = ? AND model LIKE ?",
                    [(i, f"%{annotator if annotator == 'sam3' else engine.model_ref}%") for i in keep_ids],
                )
                tags.clear_origin(write_conn, keep_ids, annotator)
            for asset_id, detections in zip(keep_ids, results):
                write_conn.executemany(
                    "INSERT INTO detections(asset_id, label, conf, x, y, w, h, model, mask) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(asset_id, d.label, d.conf, d.x, d.y, d.w, d.h, d.model, d.mask) for d in detections],
                )
                total_dets += len(detections)
                labels = sorted({f"{prefix}:{d.label}" for d in detections})
                if labels:
                    tags.add_tags(write_conn, [asset_id], labels, kind=kind, origin=annotator)
            write_conn.executemany(
                "UPDATE assets SET annotated = ? WHERE id = ?",
                [(annotator, i) for i in keep_ids],
            )
        processed += len(keep_ids)
        job.advance(len(batch_ids), f"{processed:,} assets · {total_dets:,} detections")

    return {"assets": processed, "detections": total_dets, "annotator": annotator, "prompts": prompts or []}


def _load_for_model(row) -> Image.Image | None:
    """
    Prefer the original bytes for detection — a 320px thumbnail loses the small
    fauna these models are being asked to find — and fall back to the thumbnail
    if the source object is unreachable.
    """
    try:
        backend = get_backend(_source_root(row["id"]))
        raw = backend.read_bytes(row["uri"])
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            return _downscale(opened).convert("RGB")
    except Exception:
        pass
    try:
        if row["thumb_path"]:
            with Image.open(thumb_abs_path(row["thumb_path"])) as img:
                return img.convert("RGB")
    except Exception:
        pass
    return None


def _source_root(asset_id: int) -> str:
    row = db.connect().execute(
        "SELECT s.root FROM assets a JOIN sources s ON s.id = a.source_id WHERE a.id = ?", (asset_id,)
    ).fetchone()
    return row["root"] if row else ""


def asset_bytes(asset_id: int) -> tuple[bytes, str]:
    """Fetch one asset's original bytes (used by the full-resolution viewer)."""
    row = db.connect().execute(
        "SELECT a.uri, a.ext, s.root FROM assets a JOIN sources s ON s.id = a.source_id WHERE a.id = ?",
        (asset_id,),
    ).fetchone()
    if not row:
        raise KeyError(f"No asset {asset_id}")
    backend = get_backend(row["root"])
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
    }.get(row["ext"], "application/octet-stream")
    return backend.read_bytes(row["uri"]), mime


def signed_url_for(asset_id: int) -> str | None:
    if not settings.USE_SIGNED_URLS:
        return None
    row = db.connect().execute(
        "SELECT a.uri, s.root FROM assets a JOIN sources s ON s.id = a.source_id WHERE a.id = ?",
        (asset_id,),
    ).fetchone()
    if not row:
        return None
    return get_backend(row["root"]).signed_url(row["uri"], settings.SIGNED_URL_TTL)
