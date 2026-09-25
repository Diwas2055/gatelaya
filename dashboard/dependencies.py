"""Shared route dependencies: settings, DB session, bearer-token auth."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_sessionmaker
from .settings import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_db() -> AsyncIterator[AsyncSession]:
    """Per-request database session, closed when the response is sent."""
    async with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db)]


async def require_token(request: Request, settings: SettingsDep) -> None:
    """Allow the request when no token is configured or the bearer token matches."""
    if not settings.token:
        return
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or credentials != settings.token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
