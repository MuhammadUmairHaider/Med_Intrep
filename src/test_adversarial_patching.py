import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure src is in path to import properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interpretability.feature_importance.adversarial_patching import \
    AdversarialPatchingExplainer


def main():
    print("Loading model for test...")
    model_name = "google/medgemma-1.5-4b-it"
    # Using local checkpoint if possible, else download
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "checkpoints",
            "classifier_run",
            "final_model",
        )
    )
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found, falling back to dummy/base...")
        # Since this is just a script test we can try to fall back or exit.
        return

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=20,  # 20 classes for cancer
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    explainer = AdversarialPatchingExplainer(model, tokenizer)

    test_text = "The patient presents with an invasive ductal carcinoma of the right breast. The tumor measures 2.5cm."

    print("Running Adversarial Patching interpretation...")
    results = explainer.interpret(
        input_text=test_text,
        kl_threshold=0.1,  # KL tolerance
        max_epochs=50,  # small for test
        lr=0.1,
    )

    print(f"\nOptimization Finished.")
    print(f"Target Class Idx: {results['target_class_idx']}")
    print(f"Final KL Divergence: {results['kl_divergence']:.4f}")
    print(f"Best Sparsity Rate: {results['sparsity_rate']:.4f}")
    print("\nTokens and Scores (1.0 = kept, 0.0 = patched):")

    for token, score in zip(results["tokens"], results["scores"]):
        if score > 0.5:
            print(f"KEEP: {token} (score: {score:.2f})")
        else:
            print(f"PATCH: {token} (score: {score:.2f})")


if __name__ == "__main__":
    main()
