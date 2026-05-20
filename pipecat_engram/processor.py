"""Pipecat FrameProcessor that wires Engram memory into a Pipeline.

Behavior on each finalized user turn:
    1. Store the user's text to Engram (POST /v1/buckets/{bucket}/memories).
    2. Query Engram for relevant memories (POST /v1/query).
    3. Emit a recall TextFrame downstream BEFORE the original user frame,
       so the LLM context aggregator sees the memory context first.

Frame types this processor acts on:
    - TranscriptionFrame: STT finalized user utterance (primary voice path).
    - TextFrame (only when not a subclass like LLMTextFrame/TTSTextFrame/
      InterimTranscriptionFrame): for text-input pipelines and tests.

Frames passed through unchanged:
    - All system/control frames (handled by ``super().process_frame``).
    - LLM/TTS text frames (assistant output) — we never store those.
    - Interim transcriptions — too noisy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_engram.client import EngramClient


# Sentinel for "don't store, don't recall" frame types we still want to
# forward unchanged.
_ASSISTANT_TEXT_TYPES: tuple[type, ...] = (LLMTextFrame, TTSTextFrame)


class EngramMemoryProcessor(FrameProcessor):
    """Pipecat processor backed by the Engram memory service.

    Insert this processor in the pipeline AFTER the STT (or text input)
    stage and BEFORE the LLM / context aggregator. The processor will:

    - Persist every finalized user turn to Engram under ``bucket``.
    - Inject a recall ``TextFrame`` (with ``skip_tts=True``) right before
      the user frame so the LLM aggregator picks up the memory context.

    Args:
        api_key: Engram API key. Falls back to ``ENGRAM_API_KEY``.
        bucket: Engram bucket name to read/write (default ``"default"``).
        base_url: Override REST base URL (self-hosted Engram).
        recall_prefix: String prepended to the recall TextFrame, used to
            cue the LLM that what follows is memory context.
        store: Whether to persist user turns (default True).
        recall: Whether to query + inject recall (default True).
        min_chars: Minimum user-text length to act on (default 1).
        client: Pre-built ``EngramClient`` to reuse (otherwise one is
            constructed from ``api_key`` / env).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        bucket: str = "default",
        base_url: str | None = None,
        recall_prefix: str = "Relevant memory: ",
        store: bool = True,
        recall: bool = True,
        min_chars: int = 1,
        client: EngramClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._bucket = bucket
        self._recall_prefix = recall_prefix
        self._store = store
        self._recall = recall
        self._min_chars = max(1, min_chars)
        self._client = client or EngramClient(api_key=api_key, base_url=base_url)
        self._owns_client = client is None

    @property
    def client(self) -> EngramClient:
        return self._client

    @property
    def bucket(self) -> str:
        return self._bucket

    async def cleanup(self) -> None:  # pragma: no cover - lifecycle plumbing
        if self._owns_client:
            await self._client.aclose()
        try:
            await super().cleanup()  # type: ignore[misc]
        except AttributeError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame.

        Stores user turns + injects recall, then forwards the original
        frame downstream. Non-user frames are forwarded unchanged.
        """
        await super().process_frame(frame, direction)

        # StartFrame and other system frames: just forward.
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            return

        # Only act on user-side text in the DOWNSTREAM direction.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        user_text = self._extract_user_text(frame)
        if user_text is None:
            await self.push_frame(frame, direction)
            return

        # Recall first so the recall TextFrame lands in the LLM context
        # aggregator BEFORE the user's actual utterance.
        if self._recall:
            recall_frame = await self._build_recall_frame(user_text)
            if recall_frame is not None:
                await self.push_frame(recall_frame, direction)

        # Forward the original user frame unchanged so the rest of the
        # pipeline (LLM, TTS, etc.) behaves normally.
        await self.push_frame(frame, direction)

        # Fire-and-forget the store so we don't add latency to the turn.
        if self._store:
            asyncio.create_task(self._safe_store(user_text))

    # ---- Internals --------------------------------------------------

    def _extract_user_text(self, frame: Frame) -> str | None:
        """Return the user-text payload of ``frame`` or None if we should skip it."""
        # Skip assistant output and interim transcripts.
        if isinstance(frame, _ASSISTANT_TEXT_TYPES):
            return None
        if isinstance(frame, InterimTranscriptionFrame):
            return None

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
        elif isinstance(frame, TextFrame):
            # Plain user-side TextFrame (text-input pipelines, tests).
            text = (frame.text or "").strip()
        else:
            return None

        if len(text) < self._min_chars:
            return None
        return text

    async def _build_recall_frame(self, user_text: str) -> TextFrame | None:
        try:
            result = await self._client.query_memory(user_text, bucket=self._bucket)
        except Exception as e:  # pragma: no cover - network error path
            logger.warning(f"EngramMemoryProcessor: query_memory failed: {e}")
            return None

        answer = (result or {}).get("answer")
        if not answer or not str(answer).strip():
            return None

        text = f"{self._recall_prefix}{str(answer).strip()}"
        recall = TextFrame(text=text)
        # Don't speak the recall through TTS; it's context for the LLM.
        recall.skip_tts = True
        recall.append_to_context = True
        return recall

    async def _safe_store(self, content: str) -> None:
        try:
            await self._client.store_memory(content, bucket=self._bucket)
        except Exception as e:  # pragma: no cover - network error path
            logger.warning(f"EngramMemoryProcessor: store_memory failed: {e}")
