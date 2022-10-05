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
            if not (doc_.tensor):
                doc_.load_uri_to_audio_tensor()
                model_output = self.model.transcribe(doc_.tensor)
                doc_.text = model_output["text"]
                doc_.tags["segments"] = model_output["segments"]
                doc_.tags["language"] = model_output["language"]
                doc_.tensor = None
        return docs
