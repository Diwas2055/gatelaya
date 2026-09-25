"""Shared request-text extraction: system messages + last user message."""

from __future__ import annotations

from typing import Any


def content_to_text(content: Any) -> str:
    """Flatten chat content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif (
                isinstance(part, dict)
                and part.get("type") in (None, "text", "input_text")
                and isinstance(part.get("text"), str)
            ):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def extract_scan_text(messages: Any) -> tuple[str, int | None]:
    """Return (scan_text, last_user_index): all system texts + last user text, stripped."""
    if not isinstance(messages, list) or not messages:
        return "", None
    system_parts: list[str] = []
    last_user_idx: int | None = None
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = content_to_text(msg.get("content"))
        if role == "system" and text:
            system_parts.append(text)
        elif role == "user":
            last_user_idx = idx
    last_user_text = (
        content_to_text(messages[last_user_idx].get("content"))
        if last_user_idx is not None and isinstance(messages[last_user_idx], dict)
        else ""
    )
    return "\n".join([*system_parts, last_user_text]).strip(), last_user_idx
