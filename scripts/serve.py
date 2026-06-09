#!/usr/bin/env python
"""Start the SENTINEL inference server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
    python scripts/serve.py --reload   # development hot-reload

Endpoints after startup:
    GET  http://localhost:8000/health
    POST http://localhost:8000/score/timeseries
    POST http://localhost:8000/score/audio
    WS   ws://localhost:8000/ws/timeseries/{unit_id}
    Docs http://localhost:8000/docs
"""
import argparse
import logging

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SENTINEL inference server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Hot-reload on code changes (dev only)")
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "sentinel.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
