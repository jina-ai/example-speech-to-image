# StableDiffusionExecutor
StableDiffusionExecutor uses the stable diffusion model (https://github.com/CompVis/stable-diffusion) to generate
images from text prompt.
The model is loaded using the diffusers library under the pipeline 'CompVis/stable-diffusion-v1-4'.
It is required to provide a HF authentication token with the user accepting the ToS for using the model.
The StableDiffusionExecutor expects documents with text prompt stored in `Document.text`, will generate up to
`num_images` images represented as Document objects. The Document objects contain the image data stored in the 
`Document.uri` field and are added to the query `Document.matches` field.
The `num_images` parameter can be specified as a key-value pair in `parameters` of the request.