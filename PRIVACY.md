# Privacy

The `pipecat-engram` processor sends user-turn text (transcribed speech
or text input) and memory query parameters — `content`, `query`, `bucket`,
`memory_id` — to the Engram REST API at `https://api.lumetra.io` (or the
self-hosted base URL you configured). Memories are stored under your
Engram tenant, scoped by the API key you provided.

The plugin does not collect, log, or transmit data to any third party
other than the Engram service you've explicitly authorized. It does not
read or forward audio frames, image frames, function-call results, or
any other Pipecat frame contents — only the user-text payloads of
`TranscriptionFrame` and `TextFrame` instances flowing downstream.

For Engram's own data-handling and retention policy, see
<https://lumetra.io/privacy>.
