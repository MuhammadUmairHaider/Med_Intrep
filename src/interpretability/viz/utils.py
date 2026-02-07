from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for "user visualizations"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def plot_token_importance(
    token_list: List[str],
    importance_scores: List[float],
    title: str = "Token Importance",
    save_path: Optional[str] = None,
):
    """
    Creates a bar plot of token importance scores.
    """
    df = pd.DataFrame({"Token": token_list, "Score": importance_scores})

    # Sort by absolute score for better visualization
    df["AbsScore"] = df["Score"].abs()
    df = df.sort_values("AbsScore", ascending=False).head(20)  # Top 20 tokens

    plt.figure(figsize=(10, 8))
    # Color map: Red for positive contribution, Blue for negative
    colors = ["#d62728" if x > 0 else "#1f77b4" for x in df["Score"]]

    sns.barplot(
        x="Score", y="Token", data=df, palette=colors, hue="Token", legend=False
    )
    plt.title(title)
    plt.xlabel("Attribution Score")
    plt.ylabel("Token")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_text_heatmap(
    tokens: List[str],
    scores: List[float],
    title: str = "Text Heatmap",
    save_path: Optional[str] = None,
):
    """
    Renders text with background color corresponding to importance scores.
    """
    # Normalize scores for color mapping
    scores = np.array(scores)
    max_score = np.max(np.abs(scores))
    if max_score == 0:
        max_score = 1

    norm_scores = scores / max_score

    html_content = (
        f"<h3>{title}</h3><div style='font-family: monospace; line-height: 1.5;'>"
    )

    for token, score in zip(tokens, norm_scores):
        # Red for positive (0 to 1), Blue for negative (-1 to 0)
        # Using RGBA for transparency
        if score > 0:
            color = f"rgba(255, 0, 0, {abs(score):.2f})"
        else:
            color = f"rgba(0, 0, 255, {abs(score):.2f})"

        # Clean token for display
        display_token = token.replace("Ġ", " ")

    html_content += (
        f"<span style='background-color: {color}; padding: 0 2px; margin: 0 1px; border-radius: 3px;'>"
        f"{display_token}</span>"
    )

    html_content += "</div>"

    if save_path:
        with open(save_path, "w") as f:
            f.write(html_content)


def plot_latent_space(
    embeddings: np.ndarray,
    labels: List[str],
    method: str = "t-SNE",
    save_path: Optional[str] = None,
):
    """
    Plots 2D projection of embeddings.
    """
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], "Label": labels})

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="x",
        y="y",
        hue="Label",
        data=df,
        palette="viridis",
        s=100,
        alpha=0.8,
        edgecolor="k",
    )
    plt.title(f"{method} Projection of Latent Space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_probing_accuracy(
    layers: List[int],
    accuracies: List[float],
    title: str = "Probing Accuracy per Layer",
    save_path: Optional[str] = None,
):
    """
    Line plot of probing accuracy across layers.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=layers, y=accuracies, marker="o", linewidth=2.5, color="#2ca02c")
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()
