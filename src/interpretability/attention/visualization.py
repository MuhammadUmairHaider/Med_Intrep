from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def get_attention_weights(
    model, tokenizer, text, layer_idx: int = -1, head_idx: Optional[int] = None
):
    """
    Extracts attention weights for a specific layer and head (or average across heads).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions

    layer_attention = attentions[layer_idx].cpu().squeeze(0)

    if head_idx is not None:
        attn_matrix = layer_attention[head_idx]
    else:
        attn_matrix = layer_attention.mean(dim=0)

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    return attn_matrix.float().cpu().numpy(), tokens


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
):
    """
    Plots a heatmap of the attention matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", square=True
    )
    plt.title(title)
    plt.xlabel("Key Token")
    plt.ylabel("Query Token")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()
