"""
OceanFrame Web — FastAPI entry point.

Two surfaces share one deployment:

* ``/``        — the frame analyser: upload a video or image sequence, score
                 every frame, export the keepers.
* ``/library`` — the image library: index a GCS bucket (or a local tree),
                 search it by similarity, tag it, and build datasets.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import THUMBNAIL_DIR, UPLOAD_DIR, STATIC_DIR, TEMPLATE_DIR
from library import jobs as library_jobs
from library import settings as library_settings
from routers import export, library, stream, upload

UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
library_settings.ensure_dirs()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Any job still marked running in the catalog died with the previous
    # process; say so rather than leaving a permanent phantom progress bar.
    library_jobs.mark_interrupted()
    yield


app = FastAPI(title="OceanFrame", docs_url=None, redoc_url=None, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

app.include_router(upload.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(library.router, prefix="/api")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/library")
async def library_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="library.html",
        context={"default_source": library_settings.DEFAULT_SOURCE},
    )
