import os
import re
import random
import statistics
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW



TRAIN_CONLLU     = r'train.conllu'
DEV_GOLD_CONLLU  = r'dev.conllu'
TEST_GOLD_CONLLU = r'test.conllu'


BERT_NAME = "distilbert/distilbert-base-multilingual-cased"

BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_LEN = 256
DROP_PROB = 0.10
EARLY_STOP_PATIENCE = 4
SEEDS = [42, 43, 44, 45, 46]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def parse_conllu_sentences(path: str):

    text = open(path, encoding='utf8').read().strip()
    sents = []

    for block in re.split(r'\n\n+', text):
        sent = []

        for line in block.split('\n'):
            if not line or line.startswith('#'):
                continue

            cols = line.split('\t')

            if len(cols) != 10 or '-' in cols[0] or '.' in cols[0]:
                continue

            form = cols[1]
            upos = cols[3]
            sent.append((form, upos))

        if sent:
            sents.append(sent)

    return sents


class LabelVocab:
    def __init__(self):
        self.lab2idx = {}
        self.idx2lab = {}

    def build(self, labels: List[str]):
        for lab in labels:
            if lab not in self.lab2idx:
                i = len(self.lab2idx)
                self.lab2idx[lab] = i
                self.idx2lab[i] = lab


class UPOSSentenceDataset(Dataset):

    def __init__(self, sents, upos_vocab: LabelVocab):
        self.data = []

        for sent in sents:
            tokens = [w for w, _ in sent]
            upos = [t for _, t in sent]
            upos_ids = [upos_vocab.lab2idx[t] for t in upos]
            self.data.append((tokens, upos_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_len: int):

    batch_tokens, batch_upos = zip(*batch)

    enc = tokenizer(
        list(batch_tokens),
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    upos_labels = []

    for i in range(len(batch_tokens)):
        word_ids = enc.word_ids(batch_index=i)
        up = batch_upos[i]

        aligned = []
        prev_wid = None

        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            else:
                if wid != prev_wid:
                    aligned.append(up[wid] if wid < len(up) else -100)
                else:
                    aligned.append(-100)

            prev_wid = wid

        upos_labels.append(aligned)

    upos_labels = torch.tensor(upos_labels, dtype=torch.long)

    return enc, upos_labels


class BertForUPOS(nn.Module):
    def __init__(self, bert_name: str, n_upos: int):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_name)
        h = self.bert.config.hidden_size

        self.drop = nn.Dropout(DROP_PROB)
        self.head = nn.Linear(h, n_upos)

    def forward(self, enc, upos_labels=None):
        out = self.bert(**enc)

        x = self.drop(out.last_hidden_state)  
        logits = self.head(x)                 

        loss = None

        if upos_labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                upos_labels.view(-1),
                ignore_index=-100
            )

        return loss, logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    for enc, y in tqdm(loader, desc="Eval"):
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)

        _, logits = model(enc)
        pred = logits.argmax(-1)

        mask = y != -100

        correct += ((pred == y) & mask).sum().item()
        total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def train(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = os.path.join('runs', BERT_NAME.split('/')[-1], f'seed_{seed}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n===== SEED {seed} =====\n")
    train_s = parse_conllu_sentences(TRAIN_CONLLU)
    dev_s = parse_conllu_sentences(DEV_GOLD_CONLLU)
    test_s = parse_conllu_sentences(TEST_GOLD_CONLLU)

    upos_vocab = LabelVocab()
    upos_vocab.build([upos for sent in train_s for _, upos in sent])

    
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_NAME,
        use_fast=True,
        add_prefix_space=True
    )

    if not tokenizer.is_fast:
        raise RuntimeError("Need a FAST tokenizer for word_ids() alignment.")

    train_ds = UPOSSentenceDataset(train_s, upos_vocab)
    dev_ds = UPOSSentenceDataset(dev_s, upos_vocab)
    test_ds = UPOSSentenceDataset(test_s, upos_vocab)

    tr_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LEN)
    )

    dv_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LEN)
    )

    te_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LEN)
    )

    model = BertForUPOS(
        bert_name=BERT_NAME,
        n_upos=len(upos_vocab.lab2idx)
    ).to(device)


    print("\n===== MODEL ARCHITECTURE =====")
    print(model)
    print("==============================\n")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("===== MODEL PARAMETERS =====")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("============================\n")

    best_path = os.path.join(save_dir, "best.pt")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(tr_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_dev = 0.0
    epochs_no_improve = 0

    train_losses = []
    dev_accs = []

    for ep in range(1, EPOCHS + 1):
        model.train()

        loss_sum = 0.0
        n_batches = 0

        for enc, y in tqdm(tr_loader, desc=f"Train {ep}/{EPOCHS}"):
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = model(enc, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            loss_sum += float(loss.item())
            n_batches += 1

        train_loss = loss_sum / max(1, n_batches)
        train_losses.append(train_loss)

        dev_acc = evaluate(model, dv_loader)
        dev_accs.append(dev_acc)

        print(f"Epoch {ep}: train_loss={train_loss:.4f} DevUPOS={dev_acc:.2%}")


        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {ep} "
                f"(no dev improvement for {EARLY_STOP_PATIENCE} epochs)."
            )
            break

    print(f"\n>>> Best dev UPOS = {best_dev:.2%}")

    model.load_state_dict(torch.load(best_path, map_location=device))

    test_acc = evaluate(model, te_loader)
    print(f">>> Final test UPOS = {test_acc:.2%}\n")

    # curves
    epochs_run = len(train_losses)

    loss_path = os.path.join(save_dir, "loss_curve.png")
    acc_path = os.path.join(save_dir, "acc_curve.png")

    plt.figure()
    plt.plot(range(1, epochs_run + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs_run + 1), dev_accs, label="Dev UPOS")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved curves to: {loss_path}")
    print(f"Saved curves to: {acc_path}")
    return best_dev, test_acc


if __name__ == "__main__":
    scores = [train(seed) for seed in SEEDS]
    dev_scores = [dev for dev, test in scores]
    test_scores = [test for dev, test in scores]

    print(f"\n===== RESULTS FOR {BERT_NAME} =====")
    for seed, (dev, test) in zip(SEEDS, scores):
        print(f"Seed {seed}: best dev={dev:.2%}, test={test:.2%}")
    print(f"Dev mean ± sample SD: {statistics.mean(dev_scores):.2%} ± {statistics.stdev(dev_scores):.2%}")
    print(f"Test mean ± sample SD: {statistics.mean(test_scores):.2%} ± {statistics.stdev(test_scores):.2%}")
