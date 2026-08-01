#!/usr/bin/env python3
"""
Lightweight medical Q&A fine-tune example using a small seq2seq model.
- Uses facebook/bart-base to fine-tune on a tiny in-script medical Q&A dataset (no sentencepiece required).
- Designed to run on CPU (small data, few epochs) for demonstration.

Usage (example):
  python train.py --epochs 3 --batch_size 2

Outputs saved to ./outputs/
"""

import argparse
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "facebook/bart-base"  # uses BART tokenizer (no sentencepiece)

SAMPLE_DATA = [
    {
        "question": "What are common symptoms of influenza?",
        "answer": "Common symptoms include fever, cough, sore throat, muscle aches, fatigue, and nasal congestion.",
    },
    {
        "question": "How is Type 2 diabetes typically managed?",
        "answer": "Management includes lifestyle changes (diet and exercise), blood sugar monitoring, oral medications like metformin, and sometimes insulin therapy.",
    },
    {
        "question": "When should someone seek emergency care for chest pain?",
        "answer": "Seek emergency care if chest pain is sudden or severe, or accompanied by shortness of breath, fainting, sweating, or pain spreading to the arm or jaw.",
    },
    {
        "question": "What is a common first-line treatment for bacterial throat infections?",
        "answer": "A common first-line treatment is a course of appropriate antibiotics, such as penicillin or amoxicillin, when a bacterial cause is confirmed or strongly suspected.",
    },
]


def make_inputs(questions):
    # Prefix helps models like T5 understand the task
    return [f"question: {q.strip()}" for q in questions]


def preprocess_examples(examples, tokenizer, max_input_length=256, max_target_length=64):
    inputs = make_inputs(examples["question"])
    answers = examples["answer"]

    # Tokenize inputs and targets. Use text_target when available (newer transformers).
    try:
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=False)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(answers, max_length=max_target_length, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
    except Exception:
        # Fallback for older/newer versions: use text_target param if supported
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=answers, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Lightweight medical Q&A fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model/tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # Create tiny Dataset from SAMPLE_DATA
    raw_ds = Dataset.from_list(SAMPLE_DATA)

    # Tokenize
    tokenized = raw_ds.map(lambda ex: preprocess_examples(ex, tokenizer), batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in ("input_ids", "attention_mask", "labels")], )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args (very small for demo; CPU friendly)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training (this is a tiny demo dataset; training is quick)")
    trainer.train()

    # Save model and tokenizer
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Simple inference demo
    demo_q = "What are common symptoms of influenza?"
    input_text = make_inputs([demo_q])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    gen = model.generate(**inputs, max_length=64)
    answer = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    print("\nDemo question:", demo_q)
    print("Model answer:", answer)

    # Save demo answer
    with open("outputs/demo_answer.txt", "w") as f:
        f.write("Question:\n" + demo_q + "\n\nAnswer:\n" + answer + "\n")

    print("Done. Outputs written to ./outputs/")


if __name__ == "__main__":
    main()
