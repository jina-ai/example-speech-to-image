from docarray import DocumentArray
from jina import Executor, requests


class WhisperExecutor(Executor):
    def __init__(self, model_name: str = 'base', *args, **kwargs):
        """
        WhisperExecutor receives Documents with audio data stored in `Document.uri` or `Document.tensor`, will
        convert the speech to text and will store the text output in `Document.text`.
        The model will be selected using parameter `model_name`. Available models are available at:
        https://github.com/openai/whisper#available-models-and-languages

        :param model_name: the model name used to load whisper. Available model names: https://github.com/openai/whisper#available-models-and-languages
        """
        super().__init__(*args, **kwargs)
        self.logger.info('loading model')
        import whisper
        self.model = whisper.load_model(model_name)
        self.logger.info('model loaded')

    @requests
    def transcribe(self, docs: DocumentArray, **kwargs):

        for (i, doc_) in enumerate(docs):
            model_output = self.model.transcribe(
                doc_.uri if doc_.tensor is None else doc_.tensor, task='translate', language=doc_.tags.get('language', 'English')
            )
            doc_.text = model_output['text']
            doc_.tags['segments'] = model_output['segments']
            doc_.tags['language'] = model_output['language']
        return docs
