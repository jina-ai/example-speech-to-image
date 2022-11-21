from jina import Executor, requests
from docarray import DocumentArray, Document
import torch
from diffusers import StableDiffusionPipeline


class StableDiffusionExecutor(Executor):
    def __init__(self, auth_token, **kwargs):
        """
        StableDiffusionExecutor uses the stable diffusion model (https://github.com/CompVis/stable-diffusion) to generate
        images from text prompt.
        The model is loaded using the diffusers library under the pipeline 'CompVis/stable-diffusion-v1-4'.
        It is required to provide a HF authentication token with the user accepting the ToS for using the model.
        The StableDiffusionExecutor expects documents with text prompt stored in `Document.text`, will generate up to
        `num_images` images represented as Document objects. The Document objects contain the image data stored in the
        `Document.uri` field and are added to the query `Document.matches` field.
        The `num_images` parameter can be specified as a key-value pair in `parameters` of the request.

        :param auth_token: HF auth token
        """
        super().__init__()
        self.diffusion = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            revision='fp16',
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        ).to('cuda')

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: dict, **kwargs):
        """
        Generate images from text using the StableDiffusion model.
        This endpoint expects  documents with text prompt stored in `Document.text`, will generate up to
        `num_images` images represented as Document objects. The Document objects contain the image data stored in the
        `Document.uri` field and are added to the query `Document.matches` field.
        The `num_images` parameter can be specified as a key-value pair in `parameters` of the request.
        """

        num_images = parameters.get('num_images', 1)

        for document in docs:
            self.generate_image_from_document(document, num_images)

        return docs

    def generate_image_from_document(
        self, document: Document, num_images: int
    ) -> Document:

        with torch.autocast('cuda'):
            generated_imgs = self.diffusion([document.text] * int(num_images)).images

        for img in generated_imgs:
            _generated_document = Document(
                tags={
                    'text': document.text,
                    'generator': self.__class__.__name__,
                }
            ).load_pil_image_to_datauri(img)

            document.matches.append(_generated_document)
