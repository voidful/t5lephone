from transformers import AutoProcessor, AutoModelForCTC, AutoModelForSeq2SeqLM
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import torch


class Telephonemizer:
    def __init__(self):
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        self.separator = Separator(phone="", word="", syllable="")
        self.backend = EspeakBackend(language="en-us",
                                     language_switch="remove-flags", )

    def convert_voice(self, audio_array):
        input_values = self.processor(audio_array, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0].replace(" ", "")

    def convert_text(self, text):
        phonemes = self.backend.phonemize([text], separator=self.separator)
        phonemes = phonemes[0].strip()
        return phonemes
