import gradio as gr
from docarray import Document
from jina import Client
import typer
import librosa

app = typer.Typer()


@app.command()
def main(host: str = 'localhost:54322'):

    client = Client(host=host)

    def speech_to_text(audio_file_uri, language):
        d = Document(uri=audio_file_uri)
        d.tensor = librosa.load(d.uri, sr=16_000)[0]
        d.tags['language'] = language

        docs = client.post(on='/', inputs=[d], parameters={'num_images': 2})

        for img in docs[0].matches:
            img.load_uri_to_image_tensor()

        return [docs[0].matches[i].tensor for i in range(2)] + [docs[0].text]

    gr.Interface(
        fn=speech_to_text,
        inputs=[
            gr.Audio(source='microphone', type='filepath'),
            gr.Dropdown(choices=['English', 'French', 'Arabic'], value='English', type='value')
        ],
        outputs=['image', 'image', 'text'],
    ).launch()


if __name__ == '__main__':
    app()
