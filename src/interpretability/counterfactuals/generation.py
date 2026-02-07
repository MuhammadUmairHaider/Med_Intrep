
import torch
from difflib import ndiff
from typing import List, Tuple

def compare_predictions(model, tokenizer, original_text: str, perturbed_text: str):
    """
    Compares model output for original vs perturbed text.
    """
    
    def generate(text):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    orig_output = generate(original_text)
    pert_output = generate(perturbed_text)
    
    return orig_output, pert_output

def sensitivity_check(model, tokenizer, template: str, labels: List[str]):
    """
    Checks if changing a single keyword flips the prediction.
    """
    results = []
    for label in labels:
        text = template.replace("{INSERT}", label)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Get log-probs
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :] # Last token logits
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = torch.max(probs, dim=0)
            
        pred_token = tokenizer.decode([top_idx])
        results.append({
            "input_label": label,
            "prediction": pred_token,
            "confidence": top_prob.item()
        })
        
    return results
