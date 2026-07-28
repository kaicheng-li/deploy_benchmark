"""Shared YAML configuration helpers for benchmark backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_config_path(script_file: str | Path) -> Path:
    """Return the config.yaml located beside an executable script."""
    return Path(script_file).resolve().with_name("config.yaml")


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML mapping and return it with its resolved file path."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config, path


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve a local path relative to the YAML file that declares it."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def resolve_model_path(model: dict[str, Any], config_path: str | Path) -> str:
    """Return a model's local path or its Hugging Face identifier unchanged."""
    source = model.get("source")
    path = model.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Model configuration requires a non-empty 'path'")
    if source == "local":
        return str(resolve_path(config_path, path))
    if source == "huggingface":
        return path
    raise ValueError("Model configuration 'source' must be 'local' or 'huggingface'")


def resolve_task_config(
    config: dict[str, Any], config_path: str | Path, local_path_keys: tuple[str, ...]
) -> tuple[str, dict[str, Any]]:
    """Select a task, attach its model path, and resolve declared local paths."""
    task_name, task, model = selected_task(config)
    resolved = dict(task)
    resolved["model_path"] = resolve_model_path(model, config_path)
    resolved["model_id"] = model.get("id", task_name)
    for key in local_path_keys:
        value = resolved.get(key)
        if isinstance(value, str):
            resolved[key] = str(resolve_path(config_path, value))
    return task_name, resolved


def selected_task(config: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Select the configured task and resolve its model definition by name."""
    task_name = config.get("mode")
    tasks = config.get("tasks")
    models = config.get("models")
    if not isinstance(task_name, str) or not isinstance(tasks, dict) or not isinstance(models, dict):
        raise ValueError("Configuration requires 'mode', 'tasks', and 'models' mappings")
    if task_name not in tasks:
        raise ValueError(f"Unknown configured mode: {task_name}")

    task = tasks[task_name]
    if not isinstance(task, dict):
        raise ValueError(f"Task '{task_name}' must be a mapping")
    model_name = task.get("model")
    if not isinstance(model_name, str) or model_name not in models:
        raise ValueError(f"Task '{task_name}' references an unknown model")

    model = models[model_name]
    if not isinstance(model, dict):
        raise ValueError(f"Model '{model_name}' must be a mapping")
    return task_name, task, model
