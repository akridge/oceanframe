"""HTTP surface: the routes the UI depends on."""
from __future__ import annotations

import io

import pytest
from PIL import Image


@pytest.fixture
def client(library, tree):
    from fastapi.testclient import TestClient

    from library import indexer, jobs

    job = jobs.start_job("index", {}, lambda j: indexer.index_source(j, str(tree)))
    for _ in jobs.stream(job, heartbeat=30):
        pass

    import main

    return TestClient(main.app)


def test_status_reports_backends(client):
    body = client.get("/api/library/status").json()
    assert body["stats"]["ok"] == 5
    assert body["embedder"]["backend"]
    assert {a["name"] for a in body["annotators"]} == {"yolo", "sam3"}


def test_library_page_renders(client):
    response = client.get("/library")
    assert response.status_code == 200
    assert "OceanFrame Library" in response.text


def test_search_and_asset_detail(client):
    results = client.post("/api/library/search", json={"sort": "quality", "page_size": 3}).json()
    assert results["total"] == 5

    asset_id = results["items"][0]["id"]
    detail = client.get(f"/api/library/asset/{asset_id}").json()
    assert set(detail["quality_breakdown"]) >= {"sharpness", "exposure", "contrast", "colour"}
    assert detail["similar"]

    assert client.get(f"/api/library/thumb/{asset_id}").headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/library/image/{asset_id}").status_code == 200


def test_missing_asset_is_404(client):
    assert client.get("/api/library/asset/999999").status_code == 404
    assert client.get("/api/library/thumb/999999").status_code == 404


def test_tag_round_trip(client):
    ids = [i["id"] for i in client.post("/api/library/search", json={"page_size": 2}).json()["items"]]
    client.post("/api/library/tags", json={"asset_ids": ids, "names": ["review:keep"]})
    assert client.post("/api/library/search", json={"tags": ["review:keep"]}).json()["total"] == 2

    client.post("/api/library/tags/remove", json={"asset_ids": ids, "names": ["review:keep"]})
    assert client.post("/api/library/search", json={"tags": ["review:keep"]}).json()["total"] == 0


def test_search_by_uploaded_image(client):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 150), (20, 90, 140)).save(buffer, "JPEG")
    response = client.post(
        "/api/library/search-by-image",
        files={"file": ("q.jpg", buffer.getvalue(), "image/jpeg")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["items"] and "score" in body["items"][0]


def test_annotate_without_weights_reports_why(client, monkeypatch):
    """A missing model must come back as an actionable 400, not a dead job."""
    from library.models import AnnotatorStatus

    monkeypatch.setattr(
        "library.annotate.yolo.YoloAnnotator.status",
        lambda self: AnnotatorStatus("yolo", False, "ultralytics is not installed."),
    )
    response = client.post("/api/library/annotate", json={"annotator": "yolo", "query": {}})
    assert response.status_code == 400
    assert "ultralytics" in response.json()["detail"]


def test_export_download_rejects_path_traversal(client):
    assert client.get("/api/library/exports/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_dataset_create_and_export(client):
    dataset = client.post("/api/library/datasets", json={"name": "d1", "query": {"status": "ok"}}).json()
    assert dataset["size"] == 5

    job = client.post(
        f"/api/library/datasets/{dataset['id']}/export",
        json={"kind": "csv", "include_images": False},
    ).json()
    assert job["kind"] == "export"

    # Drain the job stream so the export finishes before we look for the file.
    with client.stream("GET", f"/api/library/jobs/{job['id']}/stream") as stream:
        for _ in stream.iter_lines():
            pass
    assert client.get("/api/library/exports").json()["exports"]
