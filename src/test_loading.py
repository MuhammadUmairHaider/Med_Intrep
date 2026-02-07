
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/medgemma-1.5-4b-it"

def test_model():
    print(f"Testing loading of {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        
        # Simple inference test
        input_text = "Patient presents with a lump in the breast."
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**input_ids, max_new_tokens=20)
        print("Generation output:", tokenizer.decode(outputs[0]))
        
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, 4-bit loading might fail or be slow.")
    test_model()
