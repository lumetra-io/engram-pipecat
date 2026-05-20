"""Thin async REST wrapper around the Engram API.

Auth: Bearer eng_live_... (or test) key in the Authorization header.
Base URL defaults to https://api.lumetra.io but can be overridden for
self-hosted deployments.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.lumetra.io"


class EngramClient:
    """Bare async REST client for the Engram memory service.

    All methods are coroutines and return parsed JSON dicts (or raise
    ``httpx.HTTPStatusError`` on non-2xx responses). The user-facing
    parameter ``question`` is mapped to the REST field ``query``.

    Args:
        api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        base_url: Override for the REST base URL (e.g. self-hosted).
        timeout: Per-request timeout in seconds (default 30).
        client: Optional pre-built ``httpx.AsyncClient`` to reuse.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        key = api_key or os.environ.get("ENGRAM_API_KEY")
        if not key:
            raise ValueError(
                "Engram API key is required. Pass api_key=... or set "
                "ENGRAM_API_KEY in the environment."
            )
        self._api_key = key
        self._base_url = (base_url or os.environ.get("ENGRAM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        """Close the underlying httpx client (if we own it)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "EngramClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ---- Endpoints --------------------------------------------------

    async def store_memory(self, content: str, bucket: str = "default") -> dict[str, Any]:
        """POST /v1/buckets/{bucket}/memories — store one memory."""
        url = f"{self._base_url}/v1/buckets/{bucket}/memories"
        r = await self._client.post(url, headers=self._headers(), json={"content": content})
        r.raise_for_status()
        return _safe_json(r)

    async def query_memory(self, question: str, bucket: str = "default") -> dict[str, Any]:
        """POST /v1/query — semantic + graph retrieval.

        Note: the REST field is ``query``; we expose it as ``question`` for
        readability (matching the Engram MCP / SDK convention).
        """
        url = f"{self._base_url}/v1/query"
        r = await self._client.post(
            url,
            headers=self._headers(),
            json={"query": question, "bucket": bucket},
        )
        r.raise_for_status()
        return _safe_json(r)

    async def list_buckets(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """GET /v1/buckets — list memory buckets."""
        url = f"{self._base_url}/v1/buckets"
        r = await self._client.get(
            url,
            headers=self._headers(),
            params={"limit": limit, "offset": offset},
        )
        r.raise_for_status()
        return _safe_json(r)

    async def list_memories(self, bucket: str = "default", limit: int = 100) -> dict[str, Any]:
        """GET /v1/buckets/{bucket}/memories — list memories in a bucket."""
        url = f"{self._base_url}/v1/buckets/{bucket}/memories"
        r = await self._client.get(
            url,
            headers=self._headers(),
            params={"limit": limit},
        )
        r.raise_for_status()
        return _safe_json(r)

    async def delete_memory(self, bucket: str, memory_id: str) -> dict[str, Any]:
        """DELETE /v1/buckets/{bucket}/memories/{memory_id}."""
        url = f"{self._base_url}/v1/buckets/{bucket}/memories/{memory_id}"
        r = await self._client.delete(url, headers=self._headers())
        r.raise_for_status()
        return _safe_json(r)

    async def clear_memories(self, bucket: str) -> dict[str, Any]:
        """DELETE /v1/buckets/{bucket}/memories — clear all memories in a bucket."""
        url = f"{self._base_url}/v1/buckets/{bucket}/memories"
        r = await self._client.delete(url, headers=self._headers())
        r.raise_for_status()
        return _safe_json(r)


def _safe_json(r: httpx.Response) -> dict[str, Any]:
    """Return parsed JSON or ``{}`` on empty body."""
    if not r.content:
        return {}
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text}
