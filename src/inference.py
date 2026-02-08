import argparse
import os

import joblib
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Inference with Med-Gemma Classifier")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (containing adapter_model.bin)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/medgemma-1.5-4b-it",
        help="Base model name",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Input text for classification"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to CSV file with 'text' column for batch inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.csv",
        help="Output file for batch results",
    )

    args = parser.parse_args()

    # Load Label Encoder
    # Assumes label_encoder.joblib is in the checkpoint directory or parent
    le_path = os.path.join(args.checkpoint_dir, "label_encoder.joblib")
    if not os.path.exists(le_path):
        # Try parent directory
        le_path = os.path.join(
            os.path.dirname(args.checkpoint_dir), "label_encoder.joblib"
        )

    if not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Could not find label_encoder.joblib in {args.checkpoint_dir} or parent"
        )

    label_encoder = joblib.load(le_path)
    num_labels = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    print(f"Loaded {num_labels} classes: {class_names}")

    # Load Tokenizer
    print(f"Loading tokenizer {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model
    print(f"Loading base model {args.base_model}...")
    # Note: We need to initialize the base model with the same config as training
    # BUT, PeftModel will wrap it.
    # For sequence classification with LoRA, we typically load the base model with AutoModelForSequenceClassification
    # matching the num_labels, then load the adapter.

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

    # Load LoRA Adapter
    print(f"Loading adapter from {args.checkpoint_dir}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    def predict(text):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            predicted_label = class_names[predicted_class_id]
            confidence = probabilities[0][predicted_class_id].item()
        return predicted_label, confidence

    if args.text:
        label, conf = predict(args.text)
        print(f"\nInput: {args.text}")
        print(f"Prediction: {label} (Confidence: {conf:.4f})")

    elif args.file:
        print(f"Processing file {args.file}...")
        df = pd.read_csv(args.file)
        if "text" not in df.columns:
            raise ValueError("CSV must contain 'text' column")

        predictions = []
        confidences = []

        for i, row in df.iterrows():
            if i % 10 == 0:
                print(f"Processing {i}/{len(df)}...", end="\r")
            label, conf = predict(row["text"])
            predictions.append(label)
            confidences.append(conf)

        df["predicted_cancer_type"] = predictions
        df["confidence"] = confidences

        df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")

    else:
        print("Please provide --text or --file argument")


if __name__ == "__main__":
    main()
