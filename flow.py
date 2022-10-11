from jina import Flow
from executors.whisper import WhisperExecutor
from executors.stablediffusion import StableDiffusionExecutor
import os

hf_token = os.getenv('HF_TOKEN')
with Flow(port=54322).add(uses=WhisperExecutor).add(
    uses=StableDiffusionExecutor, uses_with={'auth_token': hf_token}
) as f:
    f.block()
