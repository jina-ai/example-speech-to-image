import os
from jina import Flow
from executors.whisper import WhisperExecutor
from executors.stablediffusion import StableDiffusionExecutor

hf_token = os.getenv('HF_TOKEN')

f = (
    Flow(port=54322)
    .add(uses=WhisperExecutor)
    .add(uses=StableDiffusionExecutor, uses_with={'auth_token': hf_token})
)

with f:
    f.block()
