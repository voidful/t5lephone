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
        phonemes = self.processor.batch_decode(predicted_ids)[0].replace(" ", "")
        for p, w in _phn2word_mapping_table.items():
            phonemes = phonemes.replace(p, w)
        return phonemes

    def convert_text(self, text):
        phonemes = self.backend.phonemize([text], separator=self.separator)
        phonemes = phonemes[0].strip()
        for p, w in _phn2word_mapping_table.items():
            phonemes = phonemes.replace(p, w)
        return phonemes


_phn2word_mapping_table = {'ð': 'T',
                           'ə': '@',
                           'ʌ': 'u',
                           'v': 'v',
                           'æ': 'a',
                           'n': 'n',
                           'd': 'd',
                           'ɪ': 'i',
                           't': 't',
                           'uː': 'U',
                           'eɪ': 'A',
                           'w': 'w',
                           'z': 'z',
                           'ɔ': 'o',
                           'f': 'f',
                           'ɔːɹ': 'Wr',
                           'ɹ': 'r',
                           'ɔː': 'W',
                           'b': 'b',
                           'aɪ': 'I',
                           'h': 'h',
                           'iː': 'E',
                           'i': 'E',
                           'm': 'm',
                           'ɜː': '3',
                           'tʃ': 'tS',
                           'ʃ': 'S',
                           's': 's',
                           'ɑːɹ': '~r',
                           'ɑː': '~',
                           'ɛɹ': 'er',
                           'ɛ': 'e',
                           'ɚ': 'er',
                           'l': 'l',
                           'oʊ': 'O',
                           'ʊ': 'U',
                           'ʊɹ': 'Ur',
                           'ŋ': 'N',
                           'oːɹ': 'Or',
                           'oː': 'O',
                           'ɡ': 'g',
                           'ɾ': 'd',
                           'p': 'p',
                           'θ': 'D',
                           'ɐ': '@',
                           'aʊ': '^U',
                           'ᵻ': 'i',
                           'j': 'j',
                           'ɪɹ': 'ir',
                           'k': 'k',
                           'əl': '@l',
                           'iə': 'E@',
                           'dʒ': 'dZ',
                           'ʒ': 'Z',
                           'ɔɪ': 'oi',
                           'ʔ': '|',
                           'n̩': 'n',
                           'aɪɚ': 'Ier',
                           'aɪə': 'I@',
                           'iːː': 'E',
                           'x': 'H',
                           'r': 'r',
                           'ɑ̃': 'A',
                           'ɡʲ': 'g',
                           }
