#!/usr/bin/env python3
"""
Lightweight dummy training script using Hugging Face transformers (DistilBERT).
Small in-memory dataset; runs on CPU; no heavy preprocessing.
"""

import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DummyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc["labels"] = labels
    return enc


def train(model_name: str, epochs: int = 1, batch_size: int = 2, lr: float = 5e-5, save_dir: str = "./model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Very small dummy dataset
    texts = [
        "I love this product",
        "This is the worst",
        "Absolutely fantastic",
        "I do not like it",
    ]
    labels = [1, 0, 1, 0]

    ds = DummyDataset(texts, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
            print(f"Epoch {epoch+1} step {step+1} loss {loss.item():.4f}")
        avg = total_loss / (step + 1)
        print(f"Epoch {epoch+1} finished. avg loss {avg:.4f}")

    # Save a tiny checkpoint
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Tiny training loop demo")
    p.add_argument("--model", default="distilbert-base-uncased", help="pretrained model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save-dir", default="./model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=args.save_dir)
