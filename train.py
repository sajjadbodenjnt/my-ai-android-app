#!/usr/bin/env python3
"""
Train a small Bengali causal LM chatbot using gpt2 (Hugging Face Transformers).
- Loads gpt2 with AutoTokenizer and AutoModelForCausalLM
- Sets tokenizer.pad_token = tokenizer.eos_token
- Uses a tiny in-memory Bengali Q&A dataset (raw_data)
- Uses DataCollatorForLanguageModeling(mlm=False) for generative training
- Saves final model and tokenizer to ./model
This script is intentionally small and CPU-friendly for demo purposes.
"""

import argparse
import os
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class BengaliQADataset(Dataset):
    """Simple Dataset that returns tokenized examples for causal LM training."""

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        # Return input_ids and attention_mask (Trainer + data_collator will handle padding)
        return {k: torch.tensor(v) for k, v in enc.items()}


def build_texts_from_raw(raw_data):
    # Format as a prompt-response pair. Keep simple: 'Q: <question>\nA: <answer>\n'
    texts = []
    for qa in raw_data:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        texts.append(f"Q: {q}\nA: {a}\n")
    return texts


def train(
    model_name: str = "gpt2",
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 5e-5,
    save_dir: str = "./model",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token is defined for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Tiny sample Bengali Q&A dataset (raw_data)
    raw_data = [
        {
            "question": "আপনি কেমন আছেন?",
            "answer": "আমি ভালো আছি, আপনাকে কেমন লাগছে?",
        },
        {
            "question": "বাংলাদেশের রাজধানী কোথায়?",
            "answer": "ঢাকা বাংলাদেশের রাজধানী।",
        },
        {
            "question": "বইয়ের সুপারিশ করবেন?",
            "answer": "আপনি যদি উপন্যাস পছন্দ করেন, তবে 'পথের পাঁচালী' পড়তে পারেন।",
        },
        {
            "question": "কীভাবে চা বানাবো?",
            "answer": "পাতা দিয়ে চা বানাতে হবে; প্রথমে জল গরম করে চা পাতা দিন এবং কিছুক্ষণ নামিয়ে নিন।",
        },
    ]

    texts = build_texts_from_raw(raw_data)

    dataset = BengaliQADataset(texts, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal language modeling
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="epoch",
        logging_steps=10,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model and tokenizer to ./model
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny Bengali causal LM (gpt2) for demo purposes")
    p.add_argument("--model", default="gpt2", help="pretrained model name or path (default: gpt2)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save-dir", default="./model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )
