from docarray import Document
from jina import Client

client = Client(port=54321)

docs = client.post(on= "/", inputs=[Document(uri='audio.wav')])

print(docs[0].text)