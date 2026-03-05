from pathlib import Path
from typing import Any, Dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(root: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return str(path)
