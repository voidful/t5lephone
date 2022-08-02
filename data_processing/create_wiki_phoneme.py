import nlp2
from datasets import load_dataset
from tqdm.auto import tqdm

from telephone import Telephonemizer

# from tqdm.contrib.concurrent import process_map  # or thread_map

telephonemizer = Telephonemizer(load_audio_model=False)

dataset = load_dataset("wikipedia", "20200501.en")
sents = []


def pdata(d):
    doc = d['text']
    return [[nlp2.clean_all(s), telephonemizer.convert_text(
        nlp2.clean_all(s).replace(" ", "").replace("ˈ", "").replace("ˌ", "").strip())] for s in doc.split("\n") if
            len(s) > 100]


for d in tqdm(dataset['train']):
    sents.extend(pdata(d))

# r = process_map(pdata, dataset['train'], max_workers=50)
# print(len(r))
nlp2.write_csv(sents, 'wiki_pt.csv')
