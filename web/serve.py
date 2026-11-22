"""Serve the dashboard and refit the transformer when a seed is chosen.

    python3 serve.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT / "src"))

from housing_tft.train import train_seed  # noqa: E402

SEED_MIN = 0
SEED_MAX = 100
BUNDLE = WEB / "model" / "seeds"
LOCK = threading.Lock()
BUSY: dict[str, int | None] = {"seed": None}


def ready_seeds() -> list[int]:
    found = [42]
    if BUNDLE.exists():
        for path in BUNDLE.iterdir():
            if path.name.isdigit() and (path / "model.json").exists() and (path / "metrics.json").exists():
                found.append(int(path.name))
    return sorted(set(found))


def bundle_ready(seed: int) -> bool:
    if seed == 42 and (WEB / "model" / "model.json").exists():
        return True
    path = BUNDLE / str(seed)
    return (path / "model.json").exists() and (path / "metrics.json").exists()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/seeds":
            self._send({"min": SEED_MIN, "max": SEED_MAX, "ready": ready_seeds(), "busy": BUSY["seed"]})
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/train":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            seed = int(body["seed"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._send({"error": "Send a seed integer."}, 400)
            return
        if seed < SEED_MIN or seed > SEED_MAX:
            self._send({"error": f"Choose a seed from {SEED_MIN} to {SEED_MAX}."}, 400)
            return
        if bundle_ready(seed):
            self._send({"ready": True, "seed": seed, "cached": True})
            return
        if not LOCK.acquire(blocking=False):
            self._send({"error": f"Seed {BUSY['seed']} is still training.", "busy": BUSY["seed"]}, 409)
            return
        BUSY["seed"] = seed
        try:
            out = BUNDLE / str(seed)
            out.mkdir(parents=True, exist_ok=True)
            summary = train_seed(seed, out)
            self._send({"ready": True, "seed": seed, "cached": False, **summary})
        except Exception as exc:
            self._send({"error": str(exc)}, 500)
        finally:
            BUSY["seed"] = None
            LOCK.release()


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    print("Housing Horizon on http://0.0.0.0:8765", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
