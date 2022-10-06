import gradio as gr
from docarray import Document
from jina import Client
client = Client(port=54322)

def speech_to_text(audio_file_uri):

    docs = client.post(on='/', inputs=[Document(uri=audio_file_uri)],  parameters={'num_images':2})

    for img in docs[0].matches:
        img.load_uri_to_image_tensor()


    return [docs[0].matches[i].tensor for i in range(2)] + [docs[0].text]
3
gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.Audio(source="microphone", type="filepath"),
    ],
    outputs=["image","image", "text"],
).launch()