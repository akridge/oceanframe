"""
Shared fixtures.

Every test runs against a throwaway ``LIB_DATA_DIR``, so the suite never touches
a real catalog and the modules that cache a connection or a memmap are reset
between tests.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A fresh, isolated library package rooted at a temp directory."""
    monkeypatch.setenv("LIB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LIB_EMBED_BACKEND", "hash")
    monkeypatch.setenv("LIB_PATH_TAG_PATTERN", r"(?P<year>\d{4})/(?P<site>[^/]+)")

    from library import settings

    # settings caches the env at import time, and the modules below capture
    # values from it, so both have to be reloaded for the temp dir to take.
    importlib.reload(settings)
    for name in ("library.db", "library.vectorstore", "library.indexer",
                 "library.search", "library.datasets", "library.tags", "library.embed"):
        module = sys.modules.get(name)
        if module:
            importlib.reload(module)

    from library import db, embed, vectorstore

    db.close()
    vectorstore.reset_store()
    embed.reset_embedder()
    db.init_db()

    yield tmp_path

    db.close()


def make_image(size=(320, 240), fill=(20, 90, 140), blobs=0, blur=0.0, scale=1.0):
    """A deterministic test image with controllable sharpness and content."""
    img = Image.new("RGB", size, fill)
    draw = ImageDraw.Draw(img)
    for i in range(blobs):
        x = 20 + (i * 47) % max(size[0] - 60, 1)
        y = 15 + (i * 31) % max(size[1] - 50, 1)
        draw.ellipse([x, y, x + 34, y + 22], fill=(250, 210, 60))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if scale != 1.0:
        img = img.point(lambda v: max(0, min(255, int(v * scale))))
    return img


@pytest.fixture
def tree(tmp_path):
    """A small folder tree of images with a known shape."""
    root = tmp_path / "survey"
    layout = {
        "2024/kaneohe": [("sharp_a.jpg", dict(blobs=6)), ("sharp_b.jpg", dict(blobs=4))],
        "2024/hanauma": [("soft.jpg", dict(blobs=5, blur=5.0))],
        "2023/kaneohe": [("dark.jpg", dict(blobs=3, scale=0.15))],
    }
    for folder, files in layout.items():
        target = root / folder
        target.mkdir(parents=True, exist_ok=True)
        for name, kwargs in files:
            make_image(**kwargs).save(target / name, quality=90)
    # A near-duplicate of sharp_a, the way burst-mode shooting produces them.
    make_image(blobs=6).filter(ImageFilter.GaussianBlur(0.3)).save(
        root / "2024/kaneohe" / "sharp_a_dup.jpg", quality=90
    )
    return root
