# OceanFrame

OceanFrame is a FastAPI app for uploading video files or image sequences, analyzing frames, and exporting results as CSV or ZIP.

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

## Dependency Reference

- `fastapi`: API framework for upload, analysis stream, and export routes.
- `uvicorn[standard]`: Production-capable ASGI server used to run FastAPI.
- `python-multipart`: Parses multipart form payloads for uploaded media files.
- `jinja2`: Renders HTML templates in `templates/`.
- `opencv-python`: Reads video frames and runs core frame-processing operations.
- `numpy`: Numeric primitives used in filtering and image/video calculations.
- `Pillow`: Creates and encodes thumbnails and output images.
- `piexif`: Writes EXIF metadata into JPEG exports when enabled.

## Script Reference

- `launch.py`: Local app launcher used by `python launch.py`.
- `main.py`: FastAPI application entrypoint and router registration.
- `docker-start.sh`: Convenience script to build/start the Docker stack and show status.
- `cloud_bootstrap.sh`: End-to-end Linux host bootstrap for Docker deploy + systemd startup.

## Endpoints

- `POST /api/upload` for a single video file.
- `POST /api/upload-images` for one or more image files.
- `GET /api/stream/{session_id}` for SSE analysis updates.
- `GET /api/thumb/{session_id}/{frame_index}` for stored thumbnails.
- `GET /api/frame/{session_id}/{frame_index}` for a full-resolution frame.
- `POST /api/csv/{session_id}` for CSV export.
- `POST /api/export/{session_id}` for ZIP export.

## Notes

- Uploaded media and generated thumbnails are stored under `uploads/` and cleaned up when sessions expire or are deleted.
- The app is currently unauthenticated and intended for trusted deployments unless additional access control is added.
