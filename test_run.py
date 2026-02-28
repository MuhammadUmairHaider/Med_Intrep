import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "src", "interpretability", "feature_importance")
    )
)
from adversarial_patching import AdversarialPatchingExplainer

model_name = "google/medgemma-1.5-4b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=20, torch_dtype=torch.bfloat16, device_map="auto"
)

explainer = AdversarialPatchingExplainer(base_model, tokenizer)

text_sample = "The patient presents with ductal carcinoma."
results = explainer.interpret(
    input_text=text_sample,
    kl_threshold=0.05,
    max_epochs=2,
    lr=0.1,
    baseline_type="unk",
)

print("Tokens:", results["tokens"])
print("Mask Scores:", results["scores"].shape)
print("Finished!")
