"""Shared configuration, validation, and reproducibility helpers."""

from __future__ import annotations

import json
import importlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# config.yaml lives next to this module (code/scripts/config.yaml); an absolute
# default means commands work from any working directory.
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration and resolve paths relative to the project root."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a top-level mapping")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration must contain a paths mapping")
    root = next(
        (
            candidate
            for candidate in (config_path.parent, *config_path.parents)
            if (candidate / "data" / "corpus.jsonl").is_file()
        ),
        Path.cwd().resolve(),
    )
    paths["root"] = str(root)
    paths["config_dir"] = str(config_path.parent)
    for key, value in list(paths.items()):
        if key not in {"config_dir", "root"} and isinstance(value, str):
            paths[key] = str((root / value).resolve())
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path value (e.g. a checkpoint) against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config["paths"]["root"]) / path).resolve()


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch/CUDA when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if value != "auto":
        return value
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path: str | Path) -> Any:
    """Load one JSON document with an explicit missing-file error."""
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_results_template(path: str | Path) -> dict[str, Any]:
    """Load the supplied results template without changing its schema."""
    template = load_json(path)
    if not isinstance(template, dict):
        raise ValueError("Results template must be a JSON object")
    return template
