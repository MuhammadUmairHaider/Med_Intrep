import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_data
from src.interpretability.attention.visualization import (
    get_attention_weights,
    plot_attention_heatmap,
)
from src.interpretability.feature_importance.integrated_gradients import LLMIsDefault
from src.interpretability.latent_analysis.clustering import compute_tsne
from src.interpretability.latent_analysis.probing import ProbingClassifier
from src.interpretability.viz.utils import (
    plot_latent_space,
    plot_text_heatmap,
    plot_token_importance,
)
from src.model_utils import get_latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run Interpretability Suite")
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/medgemma-1.5-4b-it",
        help="Model ID or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/interpretability",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for latent analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve Model ID (Base or Latest Checkpoint)
    model_id = get_latest_checkpoint(
        args.model_id,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "../../checkpoints"),
    )
    # If using absolute paths, adjustments might be needed depending on where train.py saves.
    # Assuming standard project structure: root/checkpoints
    if "checkpoints" not in model_id and os.path.exists("checkpoints"):
        model_id = get_latest_checkpoint(args.model_id, checkpoint_dir="checkpoints")

    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.gradient_checkpointing_enable()
    model.eval()

    # Load Data
    _, _, test_df = load_data("tcga_reports_valid.csv")
    samples = test_df.sample(args.num_samples, random_state=42)

    print(f"Running Interpretability Suite on {args.num_samples} samples...")

    # Pick one sample for deep dive
    sample_row = samples.iloc[0]
    text = sample_row["text"][:500]  # Truncate for speed
    print(f"Analyzing sample ID: {sample_row['patient_id']}")

    ig_wrapper = LLMIsDefault(model, tokenizer)
    prompt = f"### Instruction:\nAnalyze the report.\n\n### Input:\\n{text}\\n\\n### Response:\\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    if len(generated_ids) > 0:
        target_id = generated_ids[0].item()
        res = ig_wrapper.interpret(
            prompt, target_label_idx=target_id, n_steps=10, internal_batch_size=1
        )  # Low steps for speed testing

        # Save visualizations
        plot_token_importance(
            res["tokens"],
            res["scores"],
            title=f"Feature Importance for prediction '{res['target_token']}'",
            save_path=os.path.join(args.output_dir, "ig_bar_plot.png"),
        )
        plot_text_heatmap(
            res["tokens"],
            res["scores"],
            title=f"Attribution for '{res['target_token']}'",
            save_path=os.path.join(args.output_dir, "ig_heatmap.html"),
        )

    activations = []
    labels_kidney = []  # Dummy task: Is it KIRC vs others?

    print("Extracting activations...")

    for idx, row in samples.iterrows():
        txt = row["text"][:500]
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Last layer hidden state of last token
        # hidden_states is tuple of (batch, seq, dim)
        last_hidden = out.hidden_states[-1][:, -1, :].float().cpu().numpy()
        activations.append(last_hidden)

        # Label: 1 if KIRC (Kidney), 0 otherwise
        labels_kidney.append(1 if row["cancer_type"] == "KIRC" else 0)

    activations = np.vstack(activations)
    labels_kidney = np.array(labels_kidney)

    # Clustering
    if len(samples) >= 5:  # Need enough samples for t-SNE
        tsne_emb = compute_tsne(activations, perplexity=min(5, len(samples) - 1))
        plot_latent_space(
            tsne_emb,
            labels=["Kidney" if label == 1 else "Other" for label in labels_kidney],
            method="t-SNE",
            save_path=os.path.join(args.output_dir, "tsne_activations.png"),
        )

    # Probing (if we have mixed labels)
    if len(np.unique(labels_kidney)) > 1:
        probe = ProbingClassifier(layer_idx=32, concept_name="Is Kidney?")
        probe.train(activations, labels_kidney)
        probe.save(args.output_dir)
    else:
        print("Skipping probing: Single class in sampled batch.")

    attn_matrix, tokens = get_attention_weights(model, tokenizer, prompt, layer_idx=-1)
    plot_attention_heatmap(
        attn_matrix,
        tokens,
        title="Last Layer Attention Mean",
        save_path=os.path.join(args.output_dir, "attention_heatmap.png"),
    )

    print(f"\nInterpretability suite completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
