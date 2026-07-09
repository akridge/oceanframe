#!/usr/bin/env python3
"""
OceanFrame Web — launch server and open browser.
"""
import os
import threading
import time
import webbrowser

import uvicorn

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "80"))
OPEN_BROWSER = os.getenv("OPEN_BROWSER", "1").lower() not in {"0", "false", "no", "off"}
URL  = f"http://{HOST}:{PORT}"


def _open_browser():
    time.sleep(1.2)
    webbrowser.open(URL)


if __name__ == "__main__":
    print(f"\n  OceanFrame Web\n  Running at {URL}\n  Press Ctrl+C to stop\n")
    if OPEN_BROWSER:
        threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
