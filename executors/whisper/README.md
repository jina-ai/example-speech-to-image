# WhisperExecutor

WhisperExecutor convert audio to text using the OpenAI [Whisper model](https://github.com/openai/whisper).
WhisperExecutor receives Documents with audio data stored in `Document.uri` or `Document.tensor`, converts the 
speech to text and stores the text output in `Document.text`.
The model will be selected using parameter `model_name`. Available models are available at: https://github.com/openai/whisper#available-models-and-languages
It is also possible to perform translation of other languages into english. In this case, the language should
be specified in `Document.tags` as a key-value pair, for instance: 'language': 'French'.
List of supported langauges: https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L10.