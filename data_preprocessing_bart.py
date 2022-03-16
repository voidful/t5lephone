import argparse
import sys

from datasets import load_dataset
from itertools import groupby
import random
import numpy as np

import telephone


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="train data")
    parser.add_argument("--mask_tok", type=str, required=True, help="train data")
    parser.add_argument("--output_name", type=str, default="bart_pretrain_data")
    parser.add_argument("--mask_prob", default=0.06, type=float, help="mask lm probability")
    parser.add_argument("--worker", default=10, type=int, help="multi processing worker")
    parser.add_argument("--poisson_lam", default=3, type=int, help="poisson lambda")
    input_arg, others_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    others_arg = {
        k.replace("--", ""): v for k, v in zip(others_arg[:-1:2], others_arg[1::2])
    }
    return input_arg, others_arg


def main(arg=None):
    input_arg, others_arg = (
        parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    )
    MASKTOK = input_arg['mask_tok']
    dataset = load_dataset("text", data_files={'data': input_arg['data']})
    MASKTOK_PHONE = MASKTOK
    for p, w in telephone._phn2word_mapping_table.items():
        MASKTOK_PHONE = MASKTOK_PHONE.replace(p, w)
    print(MASKTOK, MASKTOK_PHONE)

    mask_ratio = []
    lengths = []
    for rep in range(100000):
        sent_length = random.randint(1, 1024)
        mask_length = []
        ind = 0
        while True:
            prob = random.random()
            if prob <= input_arg['mask_prob']:
                length = np.random.poisson(lam=input_arg['poisson_lam'])
                ind += length + 1
                mask_length.append(length)
            else:
                ind += 1
            if ind > sent_length:
                break
        if len(mask_length) > 0:
            lengths.append(np.mean(mask_length))
            mask_ratio.append(np.sum(mask_length) / sent_length)
        else:
            lengths.append(0)
    print(f"expectations length: {np.mean(lengths)}")
    print(f"expectations ratio: {np.mean(mask_ratio)}")

    def noisy(examples):
        try:
            target_sent = examples['text'].replace(" ", "").replace("ˈ", "").replace("ˌ", "").strip()
            input_sent = examples['text'].replace(" ", "").replace("ˈ", "").replace("ˌ", "").strip()
            input_sent = list(input_sent)
            ind = 0
            mask_lengths = []
            while True:
                if ind >= len(input_sent):
                    break
                word = input_sent[ind]
                prob = random.random()
                if prob <= input_arg['mask_prob'] and len(word) > 0:
                    length = np.random.poisson(lam=input_arg['poisson_lam'])
                    mask_lengths.append(length)
                    if length == 0:
                        input_sent.insert(ind, MASKTOK)
                        ind += 1
                    else:
                        input_sent[ind:ind + length] = [MASKTOK] * len(input_sent[ind:ind + length])
                        ind += length
                else:
                    ind += 1
                if ind >= len(input_sent):
                    break
            input_sent = "".join([k for k, _ in groupby(input_sent)])  # merge_repeat
            for p, w in telephone._phn2word_mapping_table.items():
                input_sent = input_sent.replace(p, w).replace(MASKTOK_PHONE, MASKTOK)
                target_sent = target_sent.replace(p, w).replace(MASKTOK_PHONE, MASKTOK)
            examples['input_sent'] = input_sent
            examples['target_sent'] = target_sent
            examples['mask_length'] = np.mean(mask_lengths) if len(mask_lengths) > 0 else 0.0
            examples['mask_ratio'] = ((np.sum(mask_lengths) if len(mask_lengths) > 0 else 0) / len(target_sent)) if len(
                target_sent) else 0
        except:
            pass
        return examples

    dataset = dataset.map(noisy, num_proc=input_arg['worker'])
    print(f"truth length: {np.mean(dataset['data']['mask_length'])}")
    print(f"truth ratio: {np.mean(dataset['data']['mask_ratio'])}")
    dataset['data'].to_csv(f'{input_arg["output_name"]}', columns=['input_sent', 'target_sent'], header=False,
                           index=False)


if __name__ == "__main__":
    main()
