"""
OceanFrame Web — FastAPI entry point.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import THUMBNAIL_DIR, UPLOAD_DIR, STATIC_DIR, TEMPLATE_DIR
from routers import upload, stream, export

UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OceanFrame", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

app.include_router(upload.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
app.include_router(export.router, prefix="/api")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
