import os
from jina import Flow
from executors.whisper import WhisperExecutor
from executors.stablediffusion import StableDiffusionExecutor

hf_token = os.getenv('HF_TOKEN')

f = (
    Flow(port=54322)
    .add(uses=WhisperExecutor, timeout_ready=-1, uses_with={'model_name': 'large'})
    .add(uses=StableDiffusionExecutor, uses_with={'auth_token': hf_token})
)
if __name__ == '__main__':
    with f:
        f.block()
