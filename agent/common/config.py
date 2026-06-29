from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a dictionary: {config_path}")
    return data


def load_api_keys_from_file(key_file_path: str) -> List[str]:
    path = Path(key_file_path)
    if not path.exists() or not path.is_file():
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _resolve_key_file_from_config(config_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    search_paths: List[Optional[str]] = []

    runtime_cfg = cfg.get("AGENT_RUNTIME") if isinstance(cfg.get("AGENT_RUNTIME"), dict) else {}
    embedding_cfg = cfg.get("EMBEDDING") if isinstance(cfg.get("EMBEDDING"), dict) else {}
    llm_cfg = cfg.get("LLM") if isinstance(cfg.get("LLM"), dict) else {}

    search_paths.append(runtime_cfg.get("KEY_FILE"))
    search_paths.append(embedding_cfg.get("key_file"))
    search_paths.append(llm_cfg.get("key_file"))

    for rel_path in search_paths:
        if not rel_path:
            continue
        key_path = Path(str(rel_path))
        if not key_path.is_absolute():
            key_path = Path(config_path).resolve().parent.parent / key_path
        if key_path.exists() and key_path.is_file():
            return str(key_path)
    return None


def ensure_api_key_from_config(config_path: str) -> Optional[str]:
    # Respect explicit env configuration first.
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        current = os.getenv(env_name)
        if current:
            return current

    try:
        cfg = load_config(config_path)
    except Exception:
        return None

    if not isinstance(cfg, dict):
        return None

    key_file = _resolve_key_file_from_config(config_path, cfg)
    if not key_file:
        return None

    keys = load_api_keys_from_file(key_file)
    if not keys:
        return None

    selected_key = keys[0]
    os.environ.setdefault("GEMINI_API_KEY", selected_key)
    os.environ.setdefault("GOOGLE_API_KEY", selected_key)
    return selected_key