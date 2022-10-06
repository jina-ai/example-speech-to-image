# audio2image

Create realistic AI generated images from human voice

Leveraging opan ai [whisper](https://openai.com/blog/whisper/) and [StableDiffusion](https://github.com/CompVis/stable-diffusion) 
in a cloud native application powered by [Jina](https://github.com/jina-ai/jina)


Under the hood the whisper and stable diffusion models are wrapped into [Executor](https://docs.jina.ai/fundamentals/executor/)s
that will make them self-contained microservices. Both of the microservices will be chained into a [Flow](https://docs.jina.ai/fundamentals/flow/). 
The Flow expose a gRPC endpoint which accept [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/) as input.

This is an example of a multi-modal application that can be built with [jina](https://github.com/jina-ai/jina)

## How to use it ?

* Install requirements:

```bash
pip intall -r requirements.txt
pip intall -r executors/stablediffusion/requirements.txt
pip intall -r executors/whisper/requirements.txt
```


* Start the jina FLow ( you need to get a [HF token](https://huggingface.co/docs/hub/security-tokens) and accept the [StableDiffusion terms](https://huggingface.co/spaces/stabilityai/stable-diffusion) to get the model weight. Otherwise you should provide it yourself to the Executor )

```bash
HF_TOKEN=YOUR_FH_TOKEN python flow.py
```

* Start the [gradio](https://gradio.app/) UI

```bash
python ui.py
```

* Or just talk directly to the backend with the jina [Client](https://docs.jina.ai/fundamentals/client/client/)

```python
from jina import Client
from docarray import Document
client = Client(port=54322)
docs = client.post("/", inputs=[Document(uri="audio.wav") for _ in range(1)])
for img in docs[0].matches:
    img.load_uri_to_image_tensor()

docs[0].matches.plot_image_sprites()
``` 