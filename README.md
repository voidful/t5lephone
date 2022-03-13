# Telephone

usage
```python
from telephone import Telephonemizer
from datasets import load_dataset
telephonemizer = Telephonemizer()

telephonemizer.convert_text("hello")

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
telephonemizer.convert_voice(ds[0]["audio"]["array"])
```