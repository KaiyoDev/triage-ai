"""Utility helpers used across the project."""
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
