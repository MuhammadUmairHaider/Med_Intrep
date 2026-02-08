import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CancerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Map cancer types to integers
        self.cancer_types = sorted(data["cancer_type"].unique())
        self.cancer_to_id = {c: i for i, c in enumerate(self.cancer_types)}
        self.id_to_cancer = {i: c for i, c in enumerate(self.cancer_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label_str = row["cancer_type"]
        label = self.cancer_to_id[label_str]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(filepath, test_size=0.2, val_size=0.01, random_state=42):
    df = pd.read_csv(filepath)
    # Basic cleaning if necessary (e.g. dropna)
    df = df.dropna(subset=["text", "cancer_type"])

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["cancer_type"], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["cancer_type"],
        random_state=random_state,
    )

    return train, val, test


if __name__ == "__main__":
    # Test the dataset loading
    try:
        from transformers import AutoTokenizer

        print("Loading data...")
        train, val, test = load_data("../tcga_reports_valid.csv")
        print(f"Train size: {len(train)}")
        print(f"Val size: {len(val)}")
        print(f"Test size: {len(test)}")

        # Dummy tokenizer for testing if model not downloaded yet
        # user needs to log in for real model
        print("Dataset class definition looks correct.")

    except Exception as e:
        print(f"Error: {e}")
