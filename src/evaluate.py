import argparse
import os
# Add parent directory to path to import src modules if needed
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Med-Gemma Classifier")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="tcga_reports_valid.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/medgemma-1.5-4b-it",
        help="Base model name",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    args = parser.parse_args()

    # Load Label Encoder
    le_path = os.path.join(args.model_path, "label_encoder.joblib")
    if not os.path.exists(le_path):
        # Try looking in parent/sibling dirs or just default location
        le_path = "checkpoints/classifier_run/label_encoder.joblib"  # fallback

    if not os.path.exists(le_path):
        print(
            f"Warning: label_encoder.joblib not found at {le_path}. evaluation might fail if classes don't match."
        )
        return

    label_encoder = joblib.load(le_path)
    class_names = label_encoder.classes_.tolist()
    num_labels = len(class_names)
    print(f"Loaded {num_labels} classes: {class_names}")

    # Load Data (Only Test Split)
    _, _, test_df = load_data(args.data_path)
    test_df["label"] = label_encoder.transform(test_df["cancer_type"])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding=False
        )

    test_ds = test_ds.map(preprocess_function, batched=True)

    # Load Model
    print(f"Loading base model {args.base_model}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label={i: c for i, c in enumerate(class_names)},
        label2id={c: i for i, c in enumerate(class_names)},
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)

    # Trainer for Evaluation
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        bf16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Evaluating...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)


if __name__ == "__main__":
    main()
