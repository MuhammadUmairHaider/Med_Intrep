from typing import Dict, List, Optional

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
    import matplotlib.patches as mpatches

    # Clean tokens for display
    clean_tokens = [
        t.replace("Ġ", "").replace("\u2581", "").replace("<0x0A>", "\\n").strip()
        for t in token_list
    ]
    # Handle empty tokens after cleaning
    clean_tokens = [t if t else "<space>" for t in clean_tokens]

    df = pd.DataFrame({"Token": clean_tokens, "Score": importance_scores})

    # Sort by absolute score for better visualization
    df["AbsScore"] = df["Score"].abs()

    # If the user passed more than 20 tokens, we'll take top 20 by importance
    # If they passed specific subset (like last 20), we respect it but still sort for the plot
    if len(df) > 20:
        df = df.sort_values("AbsScore", ascending=False).head(20)
    else:
        # If small number, just sort by importance for plot
        df = df.sort_values("AbsScore", ascending=False)

    # Determine Color Sign
    df["Sign"] = df["Score"].apply(lambda x: "Positive" if x > 0 else "Negative")

    # Filter out special/boring tokens
    ignore_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<space>", "", " "]
    df = df[~df["Token"].isin(ignore_tokens)]

    # Define Palette
    palette = {"Positive": "#d62728", "Negative": "#1f77b4"}

    plt.figure(figsize=(10, 8))

    # Barplot with hue mapped to Sign to avoid palette warnings
    ax = sns.barplot(
        x="Score", y="Token", data=df, hue="Sign", palette=palette, dodge=False
    )

    # Custom Legend
    red_patch = mpatches.Patch(color="#d62728", label="Positive Contribution")
    blue_patch = mpatches.Patch(color="#1f77b4", label="Negative Contribution")
    plt.legend(handles=[red_patch, blue_patch], loc="lower right")

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Attribution Score", fontsize=12)
    plt.ylabel("Token", fontsize=12)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
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
    max_abs = np.max(np.abs(scores))
    if max_abs == 0:
        max_abs = 1

    print(
        f"Stats: Min={np.min(scores):.4f}, Max={np.max(scores):.4f}, Mean={np.mean(scores):.4f}"
    )

    # Aggressive scaling for visibility: sigmoid-like or simple power
    # Using power scaling x^0.5 to boost small values
    # Sign preserves direction (positive/negative)
    norm_scores = np.sign(scores) * (np.abs(scores) / max_abs) ** 0.5

    html_content = (
        f"<div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #f9f9f9; font-family: monospace;'>"
        f"<h3 style='margin-top: 0; color: #333;'>{title}</h3>"
        # Legend
        f"<div style='display: flex; gap: 10px; margin-bottom: 15px; font-size: 0.9em; color: #444;'>"
        f"  <div style='display: flex; align-items: center;'><span style='display: inline-block; width: 15px; height: 15px; background-color: rgba(0, 0, 255, 0.6); margin-right: 5px; border-radius: 3px;'></span> Negative</div>"
        f"  <div style='display: flex; align-items: center;'><span style='display: inline-block; width: 15px; height: 15px; background-color: rgba(255, 255, 255, 1); border: 1px solid #ccc; margin-right: 5px; border-radius: 3px;'></span> Neutral</div>"
        f"  <div style='display: flex; align-items: center;'><span style='display: inline-block; width: 15px; height: 15px; background-color: rgba(255, 0, 0, 0.6); margin-right: 5px; border-radius: 3px;'></span> Positive</div>"
        f"</div>"
        f"<div style='font-family: monospace; line-height: 1.8; font-size: 1.1em; white-space: pre-wrap; background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #eee; color: #000;'>"
    )

    for token, score in zip(tokens, norm_scores):
        # Using RGBA for background transparency
        # Red for positive, Blue for negative
        if score > 0:
            color = f"rgba(255, 0, 0, {abs(score):.2f})"
        else:
            color = f"rgba(0, 0, 255, {abs(score):.2f})"

        # Robust Token Cleaning
        # 1. Replace SentencePiece ' ' (U+2581) with space
        # 2. Replace GPT 'Ġ' with space
        # 3. Replace <0x0A> with newline
        display_token = (
            token.replace(" ", " ")
            .replace("Ġ", " ")
            .replace("\u2581", " ")
            .replace("<0x0A>", "\n")
        )

        # Handle newlines explicitly for HTML
        if display_token == "\n":
            html_content += "<br/>"
        else:
            # Set color: black to ensure contrast (since bg is light/transparent)
            html_content += f"<span style='background-color: {color}; color: black; padding: 1px 1px; margin: 0; border-radius: 2px;' title='Token: {token} | Score: {score:.4f}'>{display_token}</span>"

    html_content += "</div></div>"

    if save_path:
        with open(save_path, "w") as f:
            f.write(html_content)

    try:
        from IPython.display import HTML, display

        display(HTML(html_content))
    except ImportError:
        print(html_content)  # Fallback for non-notebook environments


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


def plot_latent_space_interactive(
    embeddings: np.ndarray,
    labels: List[str],
    texts: List[str],
    title: str = "Interactive Latent Space",
    save_path: Optional[str] = None,
):
    """
    Creates an INTERACTIVE scatter plot using Plotly.
    Hovering over points shows the report text.
    """
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        print("Plotly not installed. Please install plotly for interactive plots.")
        return

    # Truncate texts for cleaner hover
    trunc_texts = [t[:200] + "..." if len(t) > 200 else t for t in texts]

    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "Label": labels,
            "Text": trunc_texts,
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Label",
        hover_data=["Text"],
        title=title,
        width=1000,
        height=800,
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(template="plotly_white")

    if save_path:
        fig.write_html(save_path)

    fig.show()
