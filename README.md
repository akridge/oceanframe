# OceanFrame

OceanFrame is a FastAPI app with two surfaces that share one deployment:

- **`/` — Frame analyser.** Upload a video or an image sequence, score every
  frame for blur, exposure and colour cast, and export the keepers as CSV or ZIP.
- **`/library` — Image library.** Index a Google Cloud Storage bucket (or a local
  tree), search it by visual similarity or text, tag it with YOLO / SAM 3, and
  build training datasets. See [docs/IMAGE_LIBRARY.md](docs/IMAGE_LIBRARY.md).

## Run locally

```bash
pip install -r requirements.txt
python launch.py
```

The app starts on `http://127.0.0.1:80` by default and opens a browser window.

## Run with Docker

```bash
docker compose up -d --build
```

The container listens on `http://localhost:80` and uses a persistent Docker volume for `uploads/`, including generated thumbnails.

For a cloud workstation or VM, use `restart: unless-stopped` from [docker-compose.yml](docker-compose.yml) so the service comes back after a machine reboot as long as Docker starts on boot.

## Host bootstrap

Use [cloud_bootstrap.sh](cloud_bootstrap.sh) on a Linux host to install Docker, clone the GitHub repo, start the compose stack, and register a systemd service that brings it back on boot.

Example:

```bash
curl -SL https://raw.githubusercontent.com/akridge/oceanframe/main/cloud_bootstrap.sh | sudo bash
```

By default, bootstrap deploys from `https://github.com/akridge/oceanframe`.

Optional overrides:

- `REPO_URL=https://github.com/akridge/oceanframe`
- `INSTALL_DIR=/opt/oceanframe`
- `BRANCH=main`
- `SERVICE_NAME=oceanframe.service`

## Image Library

Point it at a bucket and it builds a searchable catalog: thumbnails, OceanFrame
quality scores, perceptual hashes for near-duplicate collapsing, CLIP embeddings
for similarity and text search, and folder-derived tags.

```bash
# 1. index a source (a big first crawl belongs in a terminal, not a browser tab)
export LIB_SOURCE=gs://my-survey-bucket/2024
python -m library.cli doctor          # what is installed, what is missing
python -m library.cli index           # crawl, score, thumbnail, embed

# 2. tag with a model
python -m library.cli annotate --annotator yolo  --query '{"quality_min": 50}'
python -m library.cli annotate --annotator sam3  --prompts "fish,bleached coral"

# 3. search, then freeze the selection into a dataset and export it
python -m library.cli search "school of fish over sand" --limit 20
python -m library.cli dataset create reef-v1 --query '{"tags":["site:kaneohe"],"quality_min":55}'
python -m library.cli dataset export reef-v1 --kind yolo-detect
```

Or do all of it in the browser at `http://localhost/library`.

### What works without a GPU (or torch)

The library runs fully on its dependency-free hash descriptor: indexing, quality
scoring, image-to-image similarity, near-duplicate detection, tagging, datasets
and export. Installing `requirements-ml.txt` adds semantic **text** search
(CLIP) and model tagging (YOLO, SAM 3). Switching later needs no re-crawl —
`python -m library.cli embed --rebuild` keeps every thumbnail, score and tag.

SAM 3 weights are access-gated and are not downloaded automatically: request
them at <https://huggingface.co/facebook/sam3>, drop `sam3.pt` in `models/`, and
set `LIB_SAM3_MODEL`. Until then the SAM 3 annotator reports itself unavailable
with that message instead of failing a run.

### Deploy on a Google Cloud Workstation

```bash
LIB_SOURCE=gs://my-survey-bucket WITH_ML=1 ./deploy/workstation_setup.sh
```

It creates the venv, checks the workstation service account can read the bucket,
writes `~/.oceanframe.env`, registers a `systemd --user` service so the app comes
back after a restart, and prints the tunnel command. Auth uses Application
Default Credentials, so an attached service account with
`roles/storage.objectViewer` needs no key file on disk.

### Library configuration

Every setting is a `LIB_`-prefixed environment variable; the ones worth knowing:

| Variable | Default | What it does |
| --- | --- | --- |
| `LIB_SOURCE` | *(unset)* | Bucket or directory offered by default, e.g. `gs://bucket/2024` |
| `LIB_DATA_DIR` | `./library_data` | Catalog, vectors, thumbnails and exports |
| `LIB_EMBED_BACKEND` | `auto` | `clip`, `hash`, or `auto` (CLIP when importable) |
| `LIB_PATH_TAG_PATTERN` | *(unset)* | Regex whose named groups become `key:value` tags |
| `LIB_YOLO_MODEL` | `yolo11n.pt` | Any detect/segment/classify checkpoint |
| `LIB_SAM3_MODEL` | `sam3.pt` | Path to the SAM 3 weights |
| `LIB_DUPE_DISTANCE` | `6` | pHash Hamming distance that counts as a duplicate |
| `LIB_SIGNED_URLS` | `0` | Serve full-resolution images as GCS signed URLs |

`LIB_PATH_TAG_PATTERN` is what makes folder structure searchable. With

```
LIB_PATH_TAG_PATTERN='(?P<year>\d{4})/(?P<site>[^/]+)/(?P<transect>T\d+)'
```

`2024/kaneohe/T03/img_0912.jpg` gains the tags `year:2024`, `site:kaneohe` and
`transect:T03`, which are then facets, filters and dataset selectors.

## Tests

```bash
pip install pytest && python -m pytest
```

## Dependency Reference

- `fastapi`: API framework for upload, analysis stream, and export routes.
- `uvicorn[standard]`: Production-capable ASGI server used to run FastAPI.
- `python-multipart`: Parses multipart form payloads for uploaded media files.
- `jinja2`: Renders HTML templates in `templates/`.
- `opencv-python`: Reads video frames and runs core frame-processing operations.
- `numpy`: Numeric primitives used in filtering and image/video calculations.
- `Pillow`: Creates and encodes thumbnails and output images.
- `piexif`: Writes EXIF metadata into JPEG exports when enabled.
- `pydantic`: Validates request bodies for the library API.
- `google-cloud-storage`: Lists and reads `gs://` objects for the library, via ADC.

Optional, in [requirements-ml.txt](requirements-ml.txt):

- `torch`, `open_clip_torch`: CLIP embeddings, which enable semantic text search.
- `ultralytics`: YOLO detection/segmentation and SAM 3 concept segmentation.

## Script Reference

- `launch.py`: Local app launcher used by `python launch.py`.
- `main.py`: FastAPI application entrypoint and router registration.
- `docker-start.sh`: Convenience script to build/start the Docker stack and show status.
- `cloud_bootstrap.sh`: End-to-end Linux host bootstrap for Docker deploy + systemd startup.
- `deploy/workstation_setup.sh`: Google Cloud Workstation setup for the image library.
- `library/cli.py`: Batch index / embed / annotate / search / dataset commands.
- `scripts/make_demo_library.py`: Generates a synthetic survey tree to try the library on.

## Endpoints

- `POST /api/upload` for a single video file.
- `POST /api/upload-images` for one or more image files.
- `GET /api/stream/{session_id}` for SSE analysis updates.
- `GET /api/thumb/{session_id}/{frame_index}` for stored thumbnails.
- `GET /api/frame/{session_id}/{frame_index}` for a full-resolution frame.
- `POST /api/csv/{session_id}` for CSV export.
- `POST /api/export/{session_id}` for ZIP export.

Image library (`/api/library/…`):

- `GET  /status` for catalog stats, embedder and model availability.
- `POST /index`, `POST /embed`, `POST /annotate` to start background jobs.
- `GET  /jobs/{job_id}/stream` for SSE job progress.
- `POST /search`, `POST /facets`, `POST /folders`, `POST /duplicates` to query.
- `POST /search-by-image` for reverse image search from an upload.
- `GET  /asset/{id}`, `/thumb/{id}`, `/image/{id}` for one asset.
- `POST /tags`, `POST /tags/remove` to curate.
- `GET/POST /datasets`, `POST /datasets/{id}/export`, `GET /exports/{file}`.

## Notes

- Uploaded media and generated thumbnails are stored under `uploads/` and cleaned up when sessions expire or are deleted.
- The library catalog under `library_data/` is derived state: it can always be rebuilt by re-crawling, and the app never writes to the source bucket.
- The app is currently unauthenticated and intended for trusted deployments unless additional access control is added.
