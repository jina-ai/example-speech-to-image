import gradio as gr
from docarray import Document
from jina import Client
import typer

app = typer.Typer()


@app.command()
def main(host: str ='localhost:54322'):

    client = Client(host = host)

    def speech_to_text(audio_file_uri):

        docs = client.post(
            on='/', inputs=[Document(uri=audio_file_uri).load_uri_to_audio_tensor()], parameters={'num_images': 2}
        )

        for img in docs[0].matches:
            img.load_uri_to_image_tensor()

        return [docs[0].matches[i].tensor for i in range(2)] + [docs[0].text]

    gr.Interface(
        fn=speech_to_text,
        inputs=[
            gr.Audio(source='microphone', type='filepath'),
        ],
        outputs=['image', 'image', 'text'],
    ).launch()

if __name__ == '__main__':
    app()
