"""Cheap mtime polling for config/calibration/policy hot reload (no watchdog dep)."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger("gatelaya")

_TRUE_VALUES = ("1", "true", "yes", "on")


def env_flag(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable; unset falls back to `default`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def env_interval(name: str, default: float = 1.0) -> float:
    """Parse a seconds interval env var; <=0 disables polling, invalid falls back."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("GateLaya: %s=%r is not a number; using %s", name, raw, default)
        return default


def _mtime(path: str | Path) -> int | None:
    """Nanosecond mtime of a file, or None when it does not exist yet."""
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return None


class FileWatcher:
    """Rate-limited mtime poller: `poll()` is True while a tracked file differs.

    Interval <= 0 disables polling. `poll()` detects but does not advance
    mtimes; `commit()` applies the detection after a successful reload, so a
    failed reload keeps reporting True and retries on the next poll. Callers
    swap state references after a True result; plain attribute assignment is
    GIL-atomic under CPython, so concurrent hooks never observe a torn object.
    Newly-tracked paths seed silently.
    """

    def __init__(self, paths: Sequence[str | Path], interval: float) -> None:
        """Seed tracked mtimes so only post-construction changes are reported."""
        self.interval = interval
        self._mtimes: dict[str, int | None] = {str(p): _mtime(p) for p in paths}
        self._next_check = time.monotonic()
        self._detected: dict[str, int | None] = {}

    def poll(self, paths: Iterable[str | Path] | None = None) -> bool:
        """True when a tracked file differs from its last committed mtime."""
        if self.interval <= 0:
            return False
        now = time.monotonic()
        if now < self._next_check:
            return False
        self._next_check = now + self.interval
        candidates = list(paths) if paths is not None else list(self._mtimes)
        detected: dict[str, int | None] = {}
        for path in candidates:
            key = str(path)
            mtime = _mtime(path)
            if key not in self._mtimes:
                self._mtimes[key] = mtime  # newly tracked: seed, not a change
            elif mtime != self._mtimes[key]:
                detected[key] = mtime
        self._detected = detected
        return bool(detected)

    def commit(self) -> None:
        """Advance tracked mtimes for the changes `poll()` reported (post-reload)."""
        self._mtimes.update(self._detected)
        self._detected = {}


def build_watcher(
    paths: Sequence[str | Path], *, enabled: bool, interval: float
) -> FileWatcher | None:
    """Create a watcher for non-empty paths when hot reload is on and interval > 0."""
    tracked = [str(p) for p in paths if p]
    if not enabled or interval <= 0 or not tracked:
        return None
    return FileWatcher(tracked, interval)
