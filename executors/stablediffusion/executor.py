from jina import Executor, requests
from docarray import DocumentArray, Document
import torch
from diffusers import StableDiffusionPipeline


class StableDiffusionExecutor(Executor):
    """Stable diffusion Executor to generate code image out of text"""

    def __init__(self, auth_token, **kwargs):
        super().__init__()
        self.diffusion = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token
        ).to('cuda')

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: dict, **kwargs):

        num_images = parameters.get('num_images', 1)

        for document in docs:
            self.generate_image_from_document(document, num_images)

        return docs

    def generate_image_from_document(self, document: Document, num_images: int) -> Document:

        with torch.autocast("cuda"):
            generated_imgs = self.diffusion([document.text] * int(num_images))['sample']

        for img in generated_imgs:
            _generated_document = Document(
                tags={
                    'text': document.text,
                    'generator': self.__class__.__name__,
                }
            ).load_pil_image_to_datauri(img)

            document.matches.append(_generated_document)

