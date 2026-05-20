"""Engram memory plugin for Pipecat.

Exports:
    EngramClient: thin REST wrapper around the Engram API.
    EngramMemoryProcessor: FrameProcessor that auto-stores user turns
        and emits a recall TextFrame on each new user turn.
"""

from pipecat_engram.client import EngramClient
from pipecat_engram.processor import EngramMemoryProcessor

__all__ = ["EngramClient", "EngramMemoryProcessor"]
__version__ = "0.1.0"
