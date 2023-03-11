import os

import nlp2
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ByT5Tokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, \
    LongT5ForConditionalGeneration

wandb.init(project="t5lephone")

# ======================== Parameter ========================
input_csv = "data/en.train_filtered1.csv"
tokenizer_config = "google/byt5-base"
model_config = "google/byt5-base"
max_source_length = 1024
max_target_length = 200
lr = 3e-4
batch = 2
train_epoch = 20
grad_accum = 64
save_dir = "./byt5_base"


# ======================== Parameter ========================

# encode the inputs
class TextDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(input_csv, names=["inputs", "targets"])
        self.df = df
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.df["inputs"][idx],
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        targets = tokenizer(
            self.df["targets"][idx],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = targets.input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def collate_fn(self, d):
        batch = {}
        batch["input_ids"] = torch.stack([e["input_ids"] for e in d], dim=1)
        batch["attention_mask"] = torch.stack([e["attention_mask"] for e in d], dim=1)
        batch["labels"] = torch.stack([e["labels"] for e in d], dim=1)
        return batch


steps = 0
best_loss = 1000000
avg_tail32_loss = 1000000
save_steps = 100
losses = []

data = TextDataset()
loader = DataLoader(data, batch_size=batch)
accelerator = Accelerator(log_with='wandb')
tokenizer = ByT5Tokenizer.from_pretrained(tokenizer_config)

if 'byt5' in model_config:
    model = T5ForConditionalGeneration.from_pretrained(model_config)
elif 'longt5' in model_config:
    model = LongT5ForConditionalGeneration.from_pretrained(model_config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
for i in range(train_epoch):
    for batch in tqdm(loader):
        input_ids, attention_mask, labels = batch["input_ids"].squeeze(1), batch["attention_mask"].squeeze(1), batch[
            "labels"].squeeze(1)

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        losses.append(loss.item())
        accelerator.backward(loss)

        # break
        steps += 1
        if steps % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        if steps % grad_accum == grad_accum - 1:
            avg_tail32_loss = sum(losses[-32:]) / len(losses[-32:])
            print("avg_tail32_loss:", avg_tail32_loss)
            wandb.log({'avg_tail32_loss': avg_tail32_loss, 'step': steps})
            if avg_tail32_loss < best_loss:
                best_loss = avg_tail32_loss
               
            if (steps // grad_accum) % save_steps == 1: # set to 1 to check saving pipeline early
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                if accelerator.is_main_process:
                    # accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(save_dir,str(steps // grad_accum)), save_config=True, save_function=accelerator.save,
                                                    state_dict=accelerator.get_state_dict(model))
                    nlp2.write_json({'avg_tail32_loss': avg_tail32_loss, 'step': steps, 'best_loss': best_loss, 'lr': lr},
                                    os.path.join(save_dir,str(steps // grad_accum), "detail.json"))
                    print("best, saving model")
            losses = losses[-32:]
