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
    parser.add_argument("--output_name", type=str, default="bart_pretrain_data")
    parser.add_argument("--total_mask_prob", default=0.15, type=float, help="total mask lm probability")
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

    dataset = load_dataset("text", data_files={'data': input_arg['data']})
    MASKTOK = "<extra_id_"
    MASKTOK_PHONE = MASKTOK
    for p, w in telephone._phn2word_mapping_table.items():
        MASKTOK_PHONE = MASKTOK_PHONE.replace(p, w)

    mask_ratio = []
    lengths = []
    for rep in range(100000):
        sent_length = random.randint(1, 1024)
        mask_length = []
        ind = 0
        while True:
            prob = random.random()
            if prob <= input_arg['total_mask_prob'] / input_arg['poisson_lam']:
                length = np.random.poisson(lam=input_arg['poisson_lam'])
                ind += length + 1
                mask_length.append(length)
            else:
                ind += 1
            if ind > sent_length :
                break
        if len(mask_length) > 0:
            lengths.append(np.mean(mask_length))
            mask_ratio.append(np.sum(mask_length) / sent_length)
        else:
            lengths.append(0)
    print(f"expectations length: {np.mean(lengths)}")
    print(f"expectations ratio: {np.mean(mask_ratio)}")

    def noisy(examples):
        #try:
        target_sent = ""
        origin_sent = examples['text'].replace(" ", "").replace("ˈ", "").replace("ˌ", "").strip()
        input_sent = origin_sent
        input_sent = list(input_sent)
        len_input_sent = len(input_sent)
        total_masked_spans = int(len_input_sent*input_arg['total_mask_prob']/20+0.5)
        
        #print(input_sent[:10],len_input_sent,len_input_sent*input_arg['total_mask_prob'],input_arg['total_mask_prob'], total_masked_spans)
        mask_candidates = list(range(len_input_sent-30))
        
        random.shuffle(mask_candidates)
        which_idx_to_mask = []
        if total_masked_spans: 
            for m in mask_candidates:
                for e in which_idx_to_mask:
                    if abs(e-m) <= 30:
                        break
                else:
                    which_idx_to_mask.append(m)
                    if len(which_idx_to_mask) == total_masked_spans:
                        break
        else:
            examples['input_sent'] = None
            examples['target_sent'] = None
            examples['mask_length']= None
            examples['mask_ratio']= None
            return examples
        which_idx_to_mask = sorted(which_idx_to_mask, reverse = True)
        current_masked_tokens = 0
        ind = 0
        mask_lengths = []
        for ind in which_idx_to_mask:
            length = np.random.poisson(lam=input_arg['poisson_lam']) +1
            mask_lengths.append(length)
            mask_tok = f"<extra_id_{total_masked_spans-len(mask_lengths)}>"
            target_sent = mask_tok + "".join(input_sent[ind:ind + length]) + target_sent
            input_sent[ind:ind + length] = [mask_tok] * len(input_sent[ind:ind + length])

        # while True:
        #     if ind >= len(input_sent):
        #         break
        #     word = input_sent[ind]
        #     #prob = random.random()
        #     if ind <= input_arg['total_mask_prob']/ input_arg['poisson_lam']*2 and len(word) > 0:
        #         length = np.random.poisson(lam=input_arg['poisson_lam']) + 1
        #         mask_lengths.append(length)
        #         mask_tok = f"<extra_id_{len(mask_lengths)}>"
        #         target_sent += mask_tok + "".join(input_sent[ind:ind + length])
        #         input_sent[ind:ind + length] = [mask_tok] * len(input_sent[ind:ind + length])
        #         ind += length
        #     else:
        #         ind += 1
        #     if ind >= len(input_sent) or current_masked_tokens > max_masked_tokens:
        #         break
        input_sent = "".join([k for k, _ in groupby(input_sent)])  # merge_repeat
        for p, w in telephone._phn2word_mapping_table.items():
            input_sent = input_sent.replace(p, w).replace(MASKTOK_PHONE, MASKTOK)
            target_sent = target_sent.replace(p, w).replace(MASKTOK_PHONE, MASKTOK)
        examples['input_sent'] = input_sent
        examples['target_sent'] = target_sent
        examples['mask_length'] = np.mean(mask_lengths) if len(mask_lengths) > 0 else 0.0
        examples['mask_ratio'] = ((np.sum(mask_lengths) if len(mask_lengths) > 0 else 0) / len(origin_sent)) if len(
            origin_sent) else 0
        #except:
            #print("cannot parse", examples)
        return examples

    dataset = dataset.map(noisy, num_proc=input_arg['worker'])
    print(f"truth length: {np.mean([x for x in dataset['data']['mask_length'] if x is not None])}")
    print(f"truth ratio: {np.mean([x for x in dataset['data']['mask_ratio']if x is not None])}")
    df = dataset['data'].to_pandas()
    df.dropna(axis = 0, how = 'any', inplace = True) # drop samples with no masking (too short)
    df.to_csv(f'{input_arg["output_name"]}', columns=['input_sent', 'target_sent'], header=False,
                           index=False)


if __name__ == "__main__":
    main()
