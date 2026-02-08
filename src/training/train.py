import argparse
import os
# Add parent directory to path to import src modules if needed
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_data  # noqa: E402


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train Med-Gemma Classifier")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/medgemma-1.5-4b-it",
        help="Model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="tcga_reports_valid.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/classifier_run",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max training steps (for debugging)"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading data from {args.data_path}...")
    # Using load_data from src.dataset
    # Note: load_data returns raw splits. We'll combine them to fit encoder first or just use training classes.
    # Actually, simpler to load full dataframe to get all classes first.
    full_df = pd.read_csv(args.data_path)
    # Filter classes if needed, but assuming dataset is clean.

    # 20 Classes
    label_encoder = LabelEncoder()
    full_df["label"] = label_encoder.fit_transform(full_df["cancer_type"])
    class_names = label_encoder.classes_.tolist()
    num_labels = len(class_names)
    print(f"Detected {num_labels} classes: {class_names}")

    # Save label encoder for inference
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(args.output_dir, "label_encoder.joblib"))

    # Split data using the logic from load_data (train/val/test)
    # Re-implementing simplified split here or using load_data if it returns dataframes
    # load_data returns (train_df, val_df, test_df)
    train_df, val_df, test_df = load_data(args.data_path)

    # Encode labels in splits
    train_df["label"] = label_encoder.transform(train_df["cancer_type"])
    val_df["label"] = label_encoder.transform(val_df["cancer_type"])
    test_df["label"] = label_encoder.transform(test_df["cancer_type"])

    # Create HF Datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=args.max_length, padding=False
        )  # Dynamic padding via collator

    print("Tokenizing datasets...")
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)

    # Load Model (BFloat16, No Quantization)
    print(f"Loading model {args.model_name} for Sequence Classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: c for i, c in enumerate(class_names)},
        label2id={c: i for i, c in enumerate(class_names)},
        # dtype=torch.bfloat16, # Use dtype instead of torch_dtype if supported by AutoModel
        # Actually AutoModelForSequenceClassification.from_pretrained supports torch_dtype usually.
        # But warning said: `torch_dtype` is deprecated! Use `dtype` instead!
        # Maybe it's specific to the model config or something.
        # I'll try just 'dtype'.
        torch_dtype=torch.bfloat16,
        # Wait, if I change it and it breaks for older versions...
        # The error was about evaluation_strategy. The dtype was just a warning.
        # I'll leave torch_dtype for now unless it causes error. The warning is fine.
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure pad token id is set for model
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Common for Gemma/Llama
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,  # Limit total checkpoints to save space
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,  # Enable BF16
        report_to="wandb",
        run_name="medgemma-classifier",
        max_steps=args.max_steps,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    # Check for existing checkpoint to resume
    checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    print(f"Test Results: {test_results}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    label_encoder_path = os.path.join(final_model_path, "label_encoder.joblib")
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
