"""Local filesystem backend — used for dev, for mounted buckets (gcsfuse), and for tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from library.models import ObjectRef
from library.storage.base import StorageBackend


class LocalBackend(StorageBackend):
    kind = "local"

    def __init__(self, root: str) -> None:
        super().__init__(str(Path(root).expanduser().resolve()))
        self._root_path = Path(self.root)

    def exists(self) -> bool:
        return self._root_path.is_dir()

    def list_objects(self, extensions: set[str]) -> Iterator[ObjectRef]:
        lowered = {e.lower() for e in extensions}
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Skip hidden and thumbnail-cache directories in place so os.walk
            # does not descend into them at all.
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                if Path(filename).suffix.lower() not in lowered:
                    continue
                full = Path(dirpath) / filename
                try:
                    stat = full.stat()
                except OSError:
                    continue
                key = full.relative_to(self._root_path).as_posix()
                yield ObjectRef(
                    uri=full.as_uri(),
                    key=key,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    etag=f"{int(stat.st_mtime)}:{stat.st_size}",
                )

    def read_bytes(self, uri: str) -> bytes:
        return Path(self._path_for(uri)).read_bytes()

    @staticmethod
    def _path_for(uri: str) -> str:
        if uri.startswith("file://"):
            return unquote(urlparse(uri).path)
        return uri
