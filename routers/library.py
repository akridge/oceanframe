"""
Image library API — /api/library/*

Long operations (crawl, embed, annotate, export) return a job id immediately and
report progress over SSE; everything else is a normal request/response.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from library import datasets as datasets_mod
from library import db, indexer, jobs, modelcache, search, settings, tags as tags_mod
from library.annotate import annotator_statuses
from library.embed import embedder_status, get_embedder
from library.models import asset_to_dict
from library.quality import breakdown
from library.vectorstore import get_store

router = APIRouter(prefix="/library", tags=["library"])


# ── Request bodies ────────────────────────────────────────────────────────────

class SourceIn(BaseModel):
    root:  str = Field(..., description="gs://bucket/prefix or a local directory")
    label: str = ""
    tag_pattern: str | None = Field(
        None, description="Regex whose named groups become key:value tags for this source"
    )


class IndexIn(BaseModel):
    root:  str = ""
    force: bool = False
    limit: int = 0
    label: str = ""
    prune: bool = True
    tag_pattern: str | None = None


class EmbedIn(BaseModel):
    rebuild: bool = False


class AnnotateIn(BaseModel):
    annotator: str = "yolo"
    query:     dict | None = None
    asset_ids: list[int] = Field(default_factory=list)
    prompts:   list[str] = Field(default_factory=list)
    model:     str = ""
    replace:   bool = True
    limit:     int = 0


class TagIn(BaseModel):
    asset_ids: list[int] = Field(default_factory=list)
    query:     dict | None = None
    names:     list[str]
    kind:      str = "manual"


class DatasetIn(BaseModel):
    name:       str
    query:      dict | None = None
    asset_ids:  list[int] = Field(default_factory=list)
    notes:      str = ""
    split_mode: str = "by_folder"
    ratios:     dict | None = None


class ExportIn(BaseModel):
    kind:           str = "yolo-detect"
    include_images: bool = True
    conf:           float = 0.25
    labels:         list[str] = Field(default_factory=list)
    tag_prefix:     str = ""


class SplitIn(BaseModel):
    asset_ids: list[int]
    split:     str


class SavedQueryIn(BaseModel):
    name:  str
    query: dict


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
async def status() -> dict:
    db.init_db()
    embedder_state = embedder_status()
    if embedder_state.get("backend") is None:
        # First call: build the embedder so the banner can say what search modes
        # are actually live rather than "unknown".
        get_embedder()
        embedder_state = embedder_status()
    return {
        "stats":      search.stats(),
        "sources":    indexer.list_sources(),
        "embedder":   embedder_state,
        "annotators": annotator_statuses(),
        "jobs":       jobs.list_jobs(8),
        "config": {
            "default_source":  settings.DEFAULT_SOURCE,
            "page_size":       settings.PAGE_SIZE,
            "dupe_distance":   settings.DUPE_DISTANCE,
            "sam3_prompts":    settings.SAM3_PROMPTS,
            "path_tag_pattern": settings.PATH_TAG_PATTERN,
            "signed_urls":     settings.USE_SIGNED_URLS,
            "data_dir":        str(settings.DATA_DIR),
            "yolo_model":      settings.YOLO_MODEL,
            "sam3_model":      settings.SAM3_MODEL,
            "models":          modelcache.cached_models(),
        },
    }


# ── Sources & indexing ────────────────────────────────────────────────────────

@router.get("/sources")
async def get_sources() -> dict:
    db.init_db()
    return {"sources": indexer.list_sources()}


@router.post("/sources")
async def add_source(body: SourceIn) -> dict:
    db.init_db()
    source_id, root = indexer.ensure_source(body.root, body.label, body.tag_pattern)
    return {"id": source_id, "root": root}


@router.delete("/sources/{source_id}")
async def remove_source(source_id: int) -> dict:
    """Forget a source and its catalog rows.  Bucket objects are never touched."""
    with db.write() as conn:
        conn.execute("DELETE FROM assets WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    return {"ok": True}


@router.post("/index")
async def start_index(body: IndexIn) -> dict:
    db.init_db()
    root = body.root or settings.DEFAULT_SOURCE
    if not root:
        raise HTTPException(400, "No source given. Set LIB_SOURCE or pass a root.")
    if jobs.active_job("index"):
        raise HTTPException(409, "An index job is already running.")
    job = jobs.start_job(
        "index",
        {"root": root, "force": body.force, "limit": body.limit},
        lambda j: indexer.index_source(
            j, root, force=body.force, limit=body.limit, label=body.label,
            prune=body.prune, tag_pattern=body.tag_pattern,
        ),
    )
    return job.to_dict()


@router.post("/embed")
async def start_embed(body: EmbedIn) -> dict:
    db.init_db()
    if jobs.active_job("embed"):
        raise HTTPException(409, "An embed job is already running.")
    job = jobs.start_job("embed", body.model_dump(), lambda j: indexer.embed_missing(j, rebuild=body.rebuild))
    return job.to_dict()


@router.post("/annotate")
async def start_annotate(body: AnnotateIn) -> dict:
    db.init_db()
    ids = body.asset_ids or search.candidate_ids(body.query or {})
    if body.limit:
        ids = ids[: body.limit]
    if not ids:
        raise HTTPException(400, "Nothing selected to annotate.")
    if jobs.active_job("annotate"):
        raise HTTPException(409, "An annotation job is already running.")

    # Fail fast with the installer/weights message rather than starting a job
    # that dies on its first batch.
    from library.annotate import get_annotator  # noqa: PLC0415

    state = get_annotator(body.annotator, body.model or None).status()
    if not state.available:
        raise HTTPException(400, state.detail)

    job = jobs.start_job(
        "annotate",
        {"annotator": body.annotator, "count": len(ids), "prompts": body.prompts},
        lambda j: indexer.annotate_assets(
            j, ids, annotator=body.annotator, prompts=body.prompts or None,
            model_ref=body.model or None, replace=body.replace,
        ),
    )
    return job.to_dict()


# ── Jobs ──────────────────────────────────────────────────────────────────────

@router.get("/jobs")
async def get_jobs(limit: int = 20) -> dict:
    return {"jobs": jobs.list_jobs(limit)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    return job.to_dict()


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    return StreamingResponse(
        jobs.stream(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    job.cancel()
    return job.to_dict()


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/search")
async def post_search(query: dict = Body(default_factory=dict)) -> dict:
    db.init_db()
    return search.run(query)


@router.post("/facets")
async def post_facets(query: dict = Body(default_factory=dict)) -> dict:
    db.init_db()
    return search.facets(query)


@router.post("/folders")
async def post_folders(body: dict = Body(default_factory=dict)) -> dict:
    db.init_db()
    return search.folder_children(body.get("query") or {}, body.get("prefix") or "")


@router.post("/duplicates")
async def post_duplicates(body: dict = Body(default_factory=dict)) -> dict:
    db.init_db()
    return {"groups": search.duplicate_groups(body.get("query") or {}, int(body.get("limit") or 60))}


@router.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...), query: str = Query("{}")) -> dict:
    """Reverse image search: embed an uploaded file and rank the catalog by it."""
    db.init_db()
    raw = await file.read()
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except Exception as exc:
        raise HTTPException(415, f"Could not read that image: {exc}") from exc

    vector = get_embedder().embed_images([image])[0]
    try:
        base = json.loads(query) or {}
    except ValueError:
        base = {}
    return search.run({**base, "vector": vector.tolist(), "similar_to": None, "text": ""})


# ── Assets ────────────────────────────────────────────────────────────────────

@router.get("/asset/{asset_id}")
async def get_asset(asset_id: int, neighbours: int = 12) -> dict:
    db.init_db()
    conn = db.connect()
    row = conn.execute(
        """
        SELECT a.*, s.root, s.kind AS source_kind,
               (SELECT GROUP_CONCAT(t.name, char(31)) FROM asset_tags at JOIN tags t ON t.id = at.tag_id
                 WHERE at.asset_id = a.id) AS tags,
               (SELECT GROUP_CONCAT(DISTINCT d.label) FROM detections d WHERE d.asset_id = a.id) AS det_labels
          FROM assets a JOIN sources s ON s.id = a.source_id WHERE a.id = ?
        """,
        (asset_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, f"No asset {asset_id}")

    detail = asset_to_dict(row)
    detail["source_root"] = row["root"]
    detail["source_kind"] = row["source_kind"]
    detail["quality_breakdown"] = breakdown(
        row["blur"], row["brightness"], row["contrast"], row["color_cast"]
    )
    detail["detections"] = [
        {"label": d["label"], "conf": round(d["conf"], 4),
         "box": [d["x"], d["y"], d["w"], d["h"]], "model": d["model"],
         "mask": json.loads(d["mask"]) if d["mask"] else None}
        for d in conn.execute(
            "SELECT label, conf, x, y, w, h, model, mask FROM detections WHERE asset_id = ? "
            "ORDER BY conf DESC", (asset_id,))
    ]
    if neighbours and row["embed_row"] >= 0:
        detail["similar"] = search.run(
            {"similar_to": asset_id, "page_size": neighbours, "sort": "relevance"}
        )["items"]
    else:
        detail["similar"] = []
    return detail


@router.get("/thumb/{asset_id}")
async def get_thumb(asset_id: int):
    conn = db.connect()
    row = conn.execute("SELECT thumb_path FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if not row or not row["thumb_path"]:
        raise HTTPException(404, "No thumbnail for that asset")
    path = Path(indexer.thumb_abs_path(row["thumb_path"]))
    if not path.exists():
        raise HTTPException(404, "Thumbnail file is missing — re-run indexing")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/preview/{asset_id}")
async def get_preview(asset_id: int):
    """Mid-size cached JPEG for the detail viewer."""
    try:
        path = indexer.ensure_preview(asset_id)
    except KeyError:
        raise HTTPException(404, f"No asset {asset_id}") from None
    except Exception as exc:
        raise HTTPException(502, f"Could not build a preview: {exc}") from exc
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@router.get("/image/{asset_id}")
async def get_image(asset_id: int):
    """
    Full resolution.  Redirects to a GCS signed URL when signing is enabled
    (keeps the bytes off the app), otherwise proxies them.
    """
    url = indexer.signed_url_for(asset_id)
    if url:
        return RedirectResponse(url, status_code=302)
    try:
        data, mime = indexer.asset_bytes(asset_id)
    except KeyError:
        raise HTTPException(404, f"No asset {asset_id}") from None
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch the source object: {exc}") from exc
    return StreamingResponse(io.BytesIO(data), media_type=mime,
                             headers={"Cache-Control": "private, max-age=3600"})


# ── Tags ──────────────────────────────────────────────────────────────────────

@router.get("/tags")
async def get_tags() -> dict:
    db.init_db()
    return {"tags": tags_mod.list_tags(db.connect())}


@router.post("/tags")
async def add_tags(body: TagIn) -> dict:
    db.init_db()
    ids = body.asset_ids or search.candidate_ids(body.query or {})
    if not ids:
        raise HTTPException(400, "Nothing selected to tag.")
    with db.write() as conn:
        added = tags_mod.add_tags(conn, ids, body.names, kind=body.kind, origin="manual")
    return {"assets": len(ids), "pairs": added}


# POST rather than DELETE: a DELETE body is legal but is dropped by enough
# proxies and clients that it is not worth the ambiguity.
@router.post("/tags/remove")
async def remove_tags(body: TagIn) -> dict:
    db.init_db()
    ids = body.asset_ids or search.candidate_ids(body.query or {})
    if not ids:
        raise HTTPException(400, "Nothing selected.")
    with db.write() as conn:
        removed = tags_mod.remove_tags(conn, ids, body.names)
    return {"assets": len(ids), "removed": removed}


# ── Saved queries ─────────────────────────────────────────────────────────────

@router.get("/saved-queries")
async def get_saved_queries() -> dict:
    db.init_db()
    rows = db.connect().execute("SELECT * FROM saved_queries ORDER BY created_at DESC").fetchall()
    return {"queries": [
        {"id": r["id"], "name": r["name"], "query": json.loads(r["query_json"]), "created_at": r["created_at"]}
        for r in rows
    ]}


@router.post("/saved-queries")
async def save_query(body: SavedQueryIn) -> dict:
    import time  # noqa: PLC0415

    db.init_db()
    with db.write() as conn:
        conn.execute(
            "INSERT INTO saved_queries(name, query_json, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET query_json=excluded.query_json, created_at=excluded.created_at",
            (body.name, json.dumps(body.query), time.time()),
        )
    return {"ok": True, "name": body.name}


@router.delete("/saved-queries/{query_id}")
async def delete_saved_query(query_id: int) -> dict:
    with db.write() as conn:
        conn.execute("DELETE FROM saved_queries WHERE id = ?", (query_id,))
    return {"ok": True}


# ── Datasets ──────────────────────────────────────────────────────────────────

@router.get("/datasets")
async def get_datasets() -> dict:
    db.init_db()
    return {"datasets": datasets_mod.list_datasets()}


@router.post("/datasets")
async def create_dataset(body: DatasetIn) -> dict:
    db.init_db()
    try:
        return datasets_mod.create(
            body.name, query=body.query, asset_ids=body.asset_ids or None, notes=body.notes,
            split_mode=body.split_mode, ratios=body.ratios,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int) -> dict:
    db.init_db()
    try:
        return datasets_mod.describe(dataset_id)
    except KeyError:
        raise HTTPException(404, f"No dataset {dataset_id}") from None


@router.get("/datasets/{dataset_id}/items")
async def get_dataset_items(dataset_id: int, split: str = "", limit: int = 200, offset: int = 0) -> dict:
    db.init_db()
    return datasets_mod.items(dataset_id, split, limit, offset)


@router.post("/datasets/{dataset_id}/split")
async def update_split(dataset_id: int, body: SplitIn) -> dict:
    db.init_db()
    try:
        return {"updated": datasets_mod.set_split(dataset_id, body.asset_ids, body.split)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int) -> dict:
    datasets_mod.delete(dataset_id)
    return {"ok": True}


@router.post("/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: int, body: ExportIn) -> dict:
    db.init_db()
    try:
        datasets_mod.describe(dataset_id)
    except KeyError:
        raise HTTPException(404, f"No dataset {dataset_id}") from None
    job = jobs.start_job(
        "export",
        {"dataset_id": dataset_id, **body.model_dump()},
        lambda j: datasets_mod.export(
            j, dataset_id, kind=body.kind, include_images=body.include_images,
            conf=body.conf, labels=body.labels or None, tag_prefix=body.tag_prefix,
        ),
    )
    return job.to_dict()


@router.get("/exports")
async def list_exports() -> dict:
    settings.ensure_dirs()
    files = sorted(settings.EXPORT_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"exports": [
        {"file": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime,
         "url": f"/api/library/exports/{p.name}"}
        for p in files[:50]
    ]}


@router.get("/exports/{filename}")
async def download_export(filename: str):
    # Resolve inside the export dir so a crafted name cannot escape it.
    target = (settings.EXPORT_DIR / filename).resolve()
    if not str(target).startswith(str(settings.EXPORT_DIR.resolve())) or not target.is_file():
        raise HTTPException(404, "No such export")
    return FileResponse(target, media_type="application/zip", filename=target.name)


# ── Maintenance ───────────────────────────────────────────────────────────────

@router.get("/vectors")
async def vector_stats() -> dict:
    return get_store().stats()
