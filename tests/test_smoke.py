"""Smoke tests for pipecat-engram.

These tests exercise the full Pipecat Pipeline machinery via
``pipecat.tests.utils.run_test``. We use a fake EngramClient by default
so the tests don't require network access. To run the LIVE tests against
``api.lumetra.io`` set:

    ENGRAM_API_KEY=...
    PIPECAT_ENGRAM_LIVE=1

then ``pytest tests/test_smoke.py``.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import run_test

from pipecat_engram import EngramClient, EngramMemoryProcessor


# ---- Fakes ----------------------------------------------------------


class FakeEngramClient:
    """In-memory stand-in for EngramClient that records every call."""

    def __init__(self, recall_answer: str = "user prefers concise replies"):
        self.stored: list[tuple[str, str]] = []
        self.queries: list[tuple[str, str]] = []
        self.recall_answer = recall_answer

    async def store_memory(self, content: str, bucket: str = "default") -> dict:
        self.stored.append((content, bucket))
        return {"success": True, "id": "fake-id"}

    async def query_memory(self, question: str, bucket: str = "default") -> dict:
        self.queries.append((question, bucket))
        return {"success": True, "answer": self.recall_answer}

    async def list_memories(self, bucket: str = "default", limit: int = 100) -> dict:
        return {"memories": [], "total": 0, "limit": limit, "offset": 0}

    async def list_buckets(self, limit: int = 100, offset: int = 0) -> dict:
        return {"buckets": []}

    async def delete_memory(self, bucket: str, memory_id: str) -> dict:
        return {"success": True}

    async def clear_memories(self, bucket: str) -> dict:
        return {"success": True}

    async def aclose(self) -> None:
        return None


# ---- 1. Imports + REST contracts ------------------------------------


def test_imports() -> None:
    """Both public symbols import cleanly."""
    from pipecat_engram import EngramClient as C, EngramMemoryProcessor as P  # noqa: F401
    assert C is EngramClient
    assert P is EngramMemoryProcessor


# ---- 2. Pipeline-driven smoke test (TextFrame, fake client) ---------


async def _drain_pending_tasks() -> None:
    # Give the fire-and-forget store task a tick to run.
    for _ in range(10):
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_pipeline_textframe_stores_and_recalls() -> None:
    """End-to-end: push a TextFrame through a real Pipeline.

    Asserts:
        (a) EngramClient.store_memory was called with the user content.
        (b) A recall TextFrame was emitted downstream BEFORE the user frame.
        (c) The original user TextFrame is still forwarded downstream.
    """
    fake = FakeEngramClient(recall_answer="user prefers Celsius")
    processor = EngramMemoryProcessor(
        bucket="pipecat_smoke",
        recall_prefix="MEM: ",
        client=fake,  # type: ignore[arg-type]
    )

    user_frame = TextFrame(text="What's the weather like?")

    received_down, _ = await run_test(
        processor,
        frames_to_send=[user_frame],
        expected_down_frames=[TextFrame, TextFrame],  # recall, then user
    )

    # (b) recall frame first, with the prefix
    assert isinstance(received_down[0], TextFrame)
    assert received_down[0].text.startswith("MEM: ")
    assert "Celsius" in received_down[0].text
    # The recall is context for the LLM, not for TTS.
    assert received_down[0].skip_tts is True

    # (c) original user frame forwarded unchanged
    assert isinstance(received_down[1], TextFrame)
    assert received_down[1].text == "What's the weather like?"

    # (a) the store call is fire-and-forget — wait briefly.
    await _drain_pending_tasks()
    assert fake.stored == [("What's the weather like?", "pipecat_smoke")]
    assert fake.queries == [("What's the weather like?", "pipecat_smoke")]


@pytest.mark.asyncio
async def test_pipeline_transcriptionframe_acts_as_user_turn() -> None:
    """TranscriptionFrame (final STT) triggers store + recall."""
    fake = FakeEngramClient(recall_answer="user is allergic to peanuts")
    processor = EngramMemoryProcessor(
        bucket="pipecat_smoke",
        client=fake,  # type: ignore[arg-type]
    )

    tx = TranscriptionFrame(
        text="What snack should I get?",
        user_id="u1",
        timestamp="2026-05-19T00:00:00Z",
    )

    received_down, _ = await run_test(
        processor,
        frames_to_send=[tx],
        # recall TextFrame, then the original TranscriptionFrame
        expected_down_frames=[TextFrame, TranscriptionFrame],
    )

    assert isinstance(received_down[0], TextFrame)
    assert "peanuts" in received_down[0].text
    assert isinstance(received_down[1], TranscriptionFrame)
    assert received_down[1].text == "What snack should I get?"

    await _drain_pending_tasks()
    assert fake.stored == [("What snack should I get?", "pipecat_smoke")]


@pytest.mark.asyncio
async def test_pipeline_empty_recall_does_not_emit_recall_frame() -> None:
    """If the recall answer is empty/whitespace, no recall frame is emitted."""

    class EmptyClient(FakeEngramClient):
        async def query_memory(self, question, bucket="default"):
            self.queries.append((question, bucket))
            return {"success": True, "answer": ""}

    fake = EmptyClient()
    processor = EngramMemoryProcessor(bucket="b", client=fake)  # type: ignore[arg-type]

    received_down, _ = await run_test(
        processor,
        frames_to_send=[TextFrame(text="hello")],
        expected_down_frames=[TextFrame],  # just the original user frame
    )
    assert received_down[0].text == "hello"
    await _drain_pending_tasks()
    assert fake.stored == [("hello", "b")]


# ---- 3. Live REST smoke test (opt-in) -------------------------------


LIVE = os.environ.get("PIPECAT_ENGRAM_LIVE") == "1"


@pytest.mark.asyncio
@pytest.mark.skipif(not LIVE, reason="set PIPECAT_ENGRAM_LIVE=1 to run live REST checks")
async def test_live_six_endpoints() -> None:
    """Hit every REST endpoint with the live API key."""
    bucket = f"pipecat_live_{uuid.uuid4().hex[:8]}"
    async with EngramClient() as client:
        # store
        r = await client.store_memory("live smoke memory", bucket=bucket)
        assert r is not None

        # list_buckets
        b = await client.list_buckets(limit=5)
        assert "buckets" in b

        # list_memories
        m = await client.list_memories(bucket=bucket, limit=5)
        assert "memories" in m
        memories = m["memories"]
        assert len(memories) >= 1
        mem_id = memories[0]["id"]

        # query
        q = await client.query_memory("smoke memory?", bucket=bucket)
        assert "answer" in q

        # delete_memory
        await client.delete_memory(bucket, mem_id)

        # clear_memories
        await client.clear_memories(bucket)


@pytest.mark.asyncio
@pytest.mark.skipif(not LIVE, reason="set PIPECAT_ENGRAM_LIVE=1 to run live Pipeline check")
async def test_live_pipeline_end_to_end() -> None:
    """Full Pipecat Pipeline driving the real Engram REST API."""
    bucket = f"pipecat_live_{uuid.uuid4().hex[:8]}"
    async with EngramClient() as client:
        await client.store_memory(
            "Jacob's favorite color is octarine.", bucket=bucket
        )

        processor = EngramMemoryProcessor(bucket=bucket, client=client)
        received_down, _ = await run_test(
            processor,
            frames_to_send=[TextFrame(text="what is my favorite color?")],
        )

        # We should see at least the recall TextFrame + the user TextFrame.
        text_frames = [f for f in received_down if isinstance(f, TextFrame)]
        assert len(text_frames) >= 2
        # First one should be the recall (it's pushed before the user frame).
        assert text_frames[0].text.startswith("Relevant memory: ")

        await client.clear_memories(bucket)
