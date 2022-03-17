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

## preprocess
### from text
```python
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
separator = Separator(phone=" ", word="", syllable="")
backend = EspeakBackend(language="en-us",language_switch="remove-flags", )
backend.phonemize(["your text here"], separator=separator)
```
### from phone
bart

```shell
python data_preprocessing_bart.py --data ./example/example.txt --mask_tok "<extra_id_0>" --poisson_lam 20 --mask_prob 0.01  --output_name ./example/example_out_bart.csv
```

t5

```shell
python data_preprocessing_t5.py --data ./example/example.txt --poisson_lam 20 --mask_prob 0.15  --output_name ./example/example_out_t5.csv
```