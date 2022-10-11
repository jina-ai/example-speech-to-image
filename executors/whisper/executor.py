import whisper
from docarray import DocumentArray
from jina import Executor, requests

class WhisperExecutor(Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = whisper.load_model("base")

    @requests
    def transcribe(self, docs: DocumentArray, **kwargs):

        for (i, doc_) in enumerate(docs):
            model_output = self.model.transcribe(doc_.uri if doc_.tensor is None else doc_.tensor)
            doc_.text = model_output["text"]
            doc_.tags["segments"] = model_output["segments"]
            doc_.tags["language"] = model_output["language"]
        return docs
