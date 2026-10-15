"""Read the model provider, model name, and API key from the environment or .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from bill_extract.errors import ConfigurationError

DEFAULT_MODEL_NAME = "gpt-5-mini"


@dataclass(frozen=True)
class Config:
    """Runtime settings. The repr hides the API key."""

    api_key: str
    model_name: str
    model_provider: str = "openai"

    def __repr__(self) -> str:
        return (
            f"Config(model_provider={self.model_provider!r}, "
            f"model_name={self.model_name!r}, api_key='***')"
        )


def load_config(
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None = None,
) -> Config:
    """Load settings. Raise ConfigurationError before any bill file is opened.

    The project .env file, next to this package, fills blank variables. A .env
    file in the working directory does the same. Existing environment values
    win. The key is never written into an error message.
    """
    source = os.environ if environ is None else environ
    values = {str(key): "" if value is None else str(value) for key, value in source.items()}

    if env_file is not None:
        env_paths = [Path(env_file)]
    elif environ is None:
        env_paths = _default_env_files()
    else:
        env_paths = []

    for dotenv_path in env_paths:
        if not dotenv_path.is_file():
            continue
        try:
            file_values = _read_env_file(dotenv_path)
        except OSError as exc:
            raise ConfigurationError(
                "The .env file could not be read. Check the file permissions and try again."
            ) from exc
        for key, value in file_values.items():
            current = values.get(key, "")
            if current.strip() == "":
                values[key] = value

    api_key = _compact_secret(_unquote(values.get("OPENAI_API_KEY", "")))
    if not api_key:
        api_key = _compact_secret(_unquote(values.get("API_KEY", "")))
    if not api_key:
        raise ConfigurationError(
            "Set OPENAI_API_KEY in the project .env file before running bill extraction. "
            "API_KEY in that file is also accepted. The key was not found."
        )

    if "MODEL_PROVIDER" in values:
        model_provider = _unquote(values.get("MODEL_PROVIDER", "")).strip().lower()
        if not model_provider:
            raise ConfigurationError(
                "MODEL_PROVIDER is blank. Set it to openai, or remove MODEL_PROVIDER."
            )
    else:
        model_provider = "openai"
    if model_provider != "openai":
        shown = model_provider if len(model_provider) <= 40 else model_provider[:40] + "..."
        raise ConfigurationError(
            f"MODEL_PROVIDER must be openai. The value '{shown}' is not supported."
        )

    if "MODEL_NAME" in values:
        model_name = _unquote(values.get("MODEL_NAME", "")).strip()
        if not model_name:
            raise ConfigurationError(
                "MODEL_NAME is blank. Set a model name, or remove MODEL_NAME to use gpt-5-mini."
            )
    else:
        model_name = DEFAULT_MODEL_NAME

    return Config(api_key=api_key, model_name=model_name, model_provider=model_provider)


def _default_env_files() -> list[Path]:
    """Project .env next to the package, then a .env in the working directory."""
    project_env = Path(__file__).resolve().parents[2] / ".env"
    cwd_env = Path.cwd() / ".env"
    files = [project_env]
    try:
        same_file = cwd_env.resolve() == project_env.resolve()
    except OSError:
        same_file = False
    if not same_file:
        files.append(cwd_env)
    return files


def _read_env_file(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except ImportError:
        return _parse_env_file(path)
    loaded = dotenv_values(path)
    values: dict[str, str] = {}
    for key, value in loaded.items():
        if key and value is not None:
            values[str(key)] = str(value)
    return values


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote(value.strip())
    return values


def _compact_secret(value: str) -> str:
    """Drop whitespace so a wrapped .env line still yields one key."""
    return "".join(value.split())


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
