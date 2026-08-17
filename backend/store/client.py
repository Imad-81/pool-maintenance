"""
Prisma Client manager and connection lifecycle for PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional
from prisma import Prisma

log = logging.getLogger("backend.store")

# Global Prisma client instance
db = Prisma(auto_register=True)


async def connect_db() -> None:
    """Connect to PostgreSQL via Prisma if not already connected."""
    if not db.is_connected():
        log.info("Connecting to PostgreSQL via Prisma...")
        await db.connect()
        log.info("PostgreSQL connected via Prisma.")


async def disconnect_db() -> None:
    """Disconnect from PostgreSQL."""
    if db.is_connected():
        log.info("Disconnecting from PostgreSQL...")
        await db.disconnect()
        log.info("PostgreSQL disconnected.")


async def is_db_connected() -> bool:
    """Check database health by running a lightweight ping."""
    if not db.is_connected():
        return False
    try:
        await db.query_raw("SELECT 1")
        return True
    except Exception as e:
        log.warning("Database ping failed: %s", e)
        return False


async def get_db() -> AsyncIterator[Prisma]:
    """FastAPI dependency yielding the connected Prisma client."""
    if not db.is_connected():
        await db.connect()
    yield db
