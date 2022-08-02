# Telephone

## Installation

`pip install git+https://github.com/voidful/t5lephone.git`

## Release Model

| Model | link                                                 |
|------------------------|------------------------------------------------------|
| t5lephone_byt5         | https://huggingface.co/voidful/phoneme_byt5_v2       |
| t5lephone_mt5          | https://huggingface.co/voidful/phoneme-mt5           |
| t5lephone_longt5_local | https://huggingface.co/voidful/phoneme-longt5-local  |
| t5lephone_longt5_local | https://huggingface.co/voidful/phoneme-longt5-global |

## Usage

### convert text to phoneme

```python
from telephone import Telephonemizer
telephonemizer = Telephonemizer()
telephonemizer.convert_text("hello")
```

### convert speech to phoneme

```python
from telephone import Telephonemizer
telephonemizer = Telephonemizer()
from datasets import load_dataset
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
telephonemizer.convert_voice(ds[0]["audio"]["array"])
```

## Pretrain model

### data progressing

```shell
python ./data_processing/data_preprocessing_t5.py --data ./example/example.txt --poisson_lam 20 --mask_prob 0.15  --output_name ./example/example_out_t5.csv
```

### model training
modify and run
`python pretrain.py` 