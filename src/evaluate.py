
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from src.dataset import load_data
from sklearn.metrics import classification_report
import argparse

model_id = "google/medgemma-1.5-4b-it"

def evaluate(model_path):
    print(f"Loading base model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("Loading test data...")
    _, _, test_df = load_data('tcga_reports_valid.csv')
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
    
    predictions = []
    true_labels = []
    
    print("Running inference...")
    for i, row in test_df.iterrows():
        text = row['text']
        label = row['cancer_type']
        
        prompt = f"### Instruction:\nAnalyze the following medical report and classify the cancer type.\n\n### Input:\n{text}\n\n### Response:\n"
        
        # Determine strict generation kwargs to avoid lengthy outputs
        result = pipe(f"<s>{prompt}", max_new_tokens=20, return_full_text=False)
        pred_text = result[0]['generated_text'].strip()
        
        # Simple extraction logic (might need refinement based on model output style)
        # Assuming the model outputs just the class name or starts with it
        pred_class = pred_text.split('\n')[0].strip()
        
        predictions.append(pred_class)
        true_labels.append(label)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_df)}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./final_medgemma_model", help="Path to the fine-tuned model")
    args = parser.parse_args()
    
    evaluate(args.model_path)
