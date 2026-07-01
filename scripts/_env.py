"""Minimal, dependency-free ``.env`` loader for the experiment scripts.

Populates ``os.environ`` from a ``.env`` file at the repo root so API keys and
``OPENAI_BASE_URL`` (e.g. for Ollama) are picked up automatically when running
``python scripts/*.py``. Already-exported variables take precedence over the
file, and empty values are skipped so an unfilled placeholder never clobbers a
real key.
"""
from __future__ import annotations

import os
from typing import Optional


def load_dotenv(path: Optional[str] = None) -> None:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
