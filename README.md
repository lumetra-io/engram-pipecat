# pipecat-engram

Engram memory plugin for [Pipecat](https://github.com/pipecat-ai/pipecat) —
durable, explainable memory for voice and realtime AI agents.

`EngramMemoryProcessor` is a drop-in `FrameProcessor` that:

- **Stores** every finalized user turn to your Engram bucket.
- **Recalls** relevant memories on each new turn and injects them as a
  context `TextFrame` (with `skip_tts=True`) right before the user's
  utterance, so the LLM sees the memory but the TTS doesn't speak it.

## Install

```bash
pip install pipecat-engram
```

## Configure

Set your Engram API key (get one at <https://lumetra.io>):

```bash
export ENGRAM_API_KEY=eng_live_...
```

Or pass it directly to the constructor.

## Minimal Pipeline example

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

from pipecat_engram import EngramMemoryProcessor

# Memory sits between the user input (STT or text) and the LLM, so the
# LLM context aggregator sees recall TextFrames just before each user turn.
memory = EngramMemoryProcessor(
    bucket="my_agent",
    recall_prefix="Relevant memory: ",
)

pipeline = Pipeline([
    transport.input(),     # mic / WebRTC / etc.
    stt,                   # e.g. DeepgramSTTService
    memory,                # <-- EngramMemoryProcessor
    context_aggregator.user(),
    llm,                   # e.g. OpenAILLMService
    tts,                   # e.g. CartesiaTTSService
    transport.output(),
    context_aggregator.assistant(),
])

task = PipelineTask(pipeline)
await PipelineRunner().run(task)
```

For a pure text pipeline (no STT/TTS), `EngramMemoryProcessor` will also
act on plain `TextFrame`s flowing downstream from the user side.

## Direct REST access

```python
from pipecat_engram import EngramClient

async with EngramClient() as engram:
    await engram.store_memory("user prefers Celsius", bucket="prefs")
    result = await engram.query_memory("what units does the user prefer?", bucket="prefs")
    print(result["answer"])
```

`EngramClient` exposes six methods that map 1:1 to the Engram REST API:

| Method | Endpoint |
|---|---|
| `store_memory(content, bucket)` | `POST /v1/buckets/{bucket}/memories` |
| `query_memory(question, bucket)` | `POST /v1/query` |
| `list_buckets(limit, offset)` | `GET /v1/buckets` |
| `list_memories(bucket, limit)` | `GET /v1/buckets/{bucket}/memories` |
| `delete_memory(bucket, memory_id)` | `DELETE /v1/buckets/{bucket}/memories/{memory_id}` |
| `clear_memories(bucket)` | `DELETE /v1/buckets/{bucket}/memories` |

The user-facing parameter `question` is mapped to the REST field `query`.

## Frame handling

`EngramMemoryProcessor` is opinionated about which frames it acts on:

| Frame type | Action |
|---|---|
| `TranscriptionFrame` (finalized STT) | Store + recall |
| `TextFrame` (plain user text) | Store + recall |
| `LLMTextFrame`, `TTSTextFrame` | Forwarded unchanged — never stored |
| `InterimTranscriptionFrame` | Forwarded unchanged — too noisy |
| Any other frame | Forwarded unchanged |

Storing is fire-and-forget so it does not add latency to the turn.

## Self-hosted Engram

```python
EngramMemoryProcessor(base_url="https://engram.example.com")
```

Or via env: `ENGRAM_BASE_URL=https://engram.example.com`.

## License

MIT — Copyright (c) Lumetra Labs Inc.
