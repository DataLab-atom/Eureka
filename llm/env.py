from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def load_env(root: Path | None = None) -> None:
    global _LOADED
    if _LOADED:
        return

    env_path = (root or Path(__file__).resolve().parents[1]) / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)

    _LOADED = True


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    load_env()
    return os.getenv(key, default)
