"""Phidata JSON agent storage for bill conversations.

Session files stay on disk so a later question can follow the earlier one.
Secret fields and key-shaped text are removed before a file is written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bill_extract.errors import DependencyError

_SECRET_FIELDS = {
    "api_key",
    "apikey",
    "api-key",
    "openai_api_key",
    "authorization",
    "access_token",
}
_KEY_TEXT = re.compile(r"sk-[A-Za-z0-9_\-]+")


def redact_secrets(value: Any) -> Any:
    """Return a copy with secret fields removed and key-shaped text masked."""
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in _SECRET_FIELDS:
                continue
            cleaned[key] = redact_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _KEY_TEXT.sub("***", value)
    return value


def open_json_storage(dir_path: Path):
    """Open phidata's JSON file storage for agent sessions."""
    try:
        from phi.storage.agent.json import JsonFileAgentStorage
    except ImportError as exc:
        raise DependencyError(_missing_phidata_message()) from exc

    class BillJsonAgentStorage(JsonFileAgentStorage):
        """JsonFileAgentStorage that does not write secrets into session files."""

        def serialize(self, data: dict) -> str:
            return json.dumps(redact_secrets(data), ensure_ascii=False, indent=4)

    return BillJsonAgentStorage(dir_path)


def _missing_phidata_message() -> str:
    return (
        "Phidata is not installed, so questions about a bill cannot be answered. "
        "Bill extraction still works. Install phidata with: "
        "py -m pip install phidata==2.7.10 --no-deps"
    )
