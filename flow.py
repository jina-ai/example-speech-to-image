from jina import Flow
from executors.whisper_exec import WhisperExecutor

with Flow(port=54321).add(uses=WhisperExecutor) as f:
    f.block()