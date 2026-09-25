"""GateLaya exception hierarchy."""


class GateLayaError(Exception):
    """Base error for all GateLaya failures."""


class LayaNotInstalledError(GateLayaError):
    """Raised when the optional `laya` package cannot be imported."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Laya is not installed. Install it with: "
                "pip install 'gatelaya[model]' (or: pip install laya)"
            )
        )


class GuardrailConfigurationError(GateLayaError):
    """Raised when GateLaya configuration values are missing or invalid."""
