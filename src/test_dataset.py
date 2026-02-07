
import os
import torch
from transformers import AutoTokenizer
from src.dataset import load_data, CancerDataset
import pandas as pd

model_id = "google/medgemma-1.5-4b-it"

def test_dataset():
    print("Testing dataset creation...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        train_df, val_df, test_df = load_data('tcga_reports_valid.csv')
        print(f"Data numbers: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        train_dataset = CancerDataset(train_df, tokenizer)
        
        # Test getting one item
        sample = train_dataset[0]
        print("Sample keys:", sample.keys())
        print("Input IDs shape:", sample['input_ids'].shape)
        print("Labels:", sample['labels'])
        print("Decoded input:", tokenizer.decode(sample['input_ids'][:20]))
        
        print("Dataset test passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        raise e

if __name__ == "__main__":
    test_dataset()
