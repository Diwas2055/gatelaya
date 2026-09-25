"""Fallback CustomGuardrail base used ONLY when litellm is not importable.

CLEARLY MARKED STUB: this is a minimal stand-in so GateLaya can be imported and
unit-tested in environments without litellm. When litellm is installed,
`gatelaya.guardrail` imports the real base class and this module is never used.
"""

from __future__ import annotations

from typing import Any


class CustomGuardrail:
    """Minimal stand-in matching the litellm CustomGuardrail surface GateLaya uses."""

    def __init__(
        self,
        guardrail_name: str | None = None,
        supported_event_hooks: list[str] | None = None,
        event_hook: Any = None,
        default_on: bool = False,
        mask_request_content: bool = False,
        mask_response_content: bool = False,
        **kwargs: Any,
    ) -> None:
        self.guardrail_name = guardrail_name
        self.supported_event_hooks = supported_event_hooks
        self.event_hook = event_hook
        self.default_on = default_on
        self.mask_request_content = mask_request_content
        self.mask_response_content = mask_response_content

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str
    ) -> Any:
        """Stub pre-call hook (real behavior lives in GateLayaGuardrail)."""
        raise NotImplementedError("stub CustomGuardrail; install litellm for the real base class")

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        """Stub post-call hook (real behavior lives in GateLayaGuardrail)."""
        raise NotImplementedError("stub CustomGuardrail; install litellm for the real base class")

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> Any:
        """Stub streaming hook (real behavior lives in GateLayaGuardrail)."""
        raise NotImplementedError("stub CustomGuardrail; install litellm for the real base class")
