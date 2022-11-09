import os
from jina import Flow
from executors.whisper import WhisperExecutor
from executors.stablediffusion import StableDiffusionExecutor

hf_token = os.getenv('HF_TOKEN')

f = (
    Flow(port=54322)
    .add(uses=WhisperExecutor)
)

with f:
    f.block()
