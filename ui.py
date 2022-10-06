import gradio as gr
from docarray import Document
from jina import Client
client = Client(port=54322)

def speech_to_text(audio_file_uri):

    docs = client.post(on='/', inputs=[Document(uri=audio_file_uri)])
    doc_generated = docs[0].matches[0]
    doc_generated.load_uri_to_image_tensor()
    return doc_generated.tensor

gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.Audio(source="microphone", type="filepath"),
    ],
    outputs="image",
).launch()