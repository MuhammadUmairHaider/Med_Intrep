import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "src", "interpretability", "feature_importance")
    )
)
from adversarial_patching import AdversarialPatchingExplainer

model_name = "google/medgemma-1.5-4b-it"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=20, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Done.")

explainer = AdversarialPatchingExplainer(model, tokenizer)

# A decently long text
text = (
    "The patient presents with an infiltrating ductal carcinoma of the right breast, Nottingham Grade 3. "
    * 20
)
print(f"Testing text length: {len(tokenizer(text).input_ids)} tokens")

print("Running optimization...")
start = time.time()
results = explainer.interpret(
    input_text=text,
    kl_threshold=0.1,
    max_epochs=10,  # Just run 10 epochs
    lr=0.1,
    baseline_type="unk",
)
end = time.time()

print(f"Time for 10 epochs: {end - start:.2f} seconds")
print(f"Projected time for 500 epochs: {(end - start) * 50:.2f} seconds")
