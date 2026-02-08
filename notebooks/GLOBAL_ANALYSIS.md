# Deep Dive: Advanced Global Interpretability

This document details the methodologies implemented in `global_analysis.ipynb`. The notebook leverages advanced interpretability techniques to understand the model's behavior not just on single examples, but on a global scale.

## 1. Global Feature Importance (Aggregated Integrated Gradients)

**Goal:** Identify which words are consistently the most predictive "signatures" for each cancer type across the entire dataset.

**Methodology:**

1.  **Token Attribution**: We use **Integrated Gradients (IG)**, a gradient-based attribution method. IG computes the integral of gradients of the model's output with respect to the input along a path from a baseline (zero embedding) to the input. This satisfies axioms like _Completeness_ (sum of attributions equals the difference in output score).
    - _Implementation_: `src/interpretability/feature_importance/integrated_gradients_classifier.py`
2.  **Aggregation**: Instead of analyzing one report, we run IG on a large subset of validation examples for a specific class (e.g., "Breast Invasive Carcinoma").
3.  **Token Alignment**: For each unique token in the vocabulary, we collect all the attribution scores it received across all documents where it appeared.
4.  **Metric**: We compute the **Mean Attribution Score** for each token.
    - **High Positive Mean**: The token consistently increases the probability of the target class.
    - **High Negative Mean**: The token consistently decreases the probability of the target class (evidence _against_ it).
    - _Implementation_: `GlobalExplainer.accumulate_token_importance` in `src/interpretability/feature_importance/global_importance.py`

**Why this matters:**
Single-example explanations can be noisy or specific to that one document. By aggregating, we filter out noise and find the true signal—the medical terms the model actually relies on.

## 2. Interactive Latent Space Visualization

**Goal:** Understand how the model internally represents and clusters different cancer types.

**Methodology:**

1.  **Embedding Extraction**: We extract the **hidden states** from the final layer of the base model (before the classification head). Specifically, we use the embedding of the last token (which acts as the CLS/aggregate representation in causal LMs for classification).
    - _Note_: The embeddings are cast to `float32` to ensure compatibility with scientific libraries (fixing `bfloat16` issues).
2.  **Dimensionality Reduction**: We use **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to project these high-dimensional vectors (e.g., 2048 dimensions) into a 2D space. t-SNE is particularly good at preserving local structure and revealing clusters.
3.  **Interactive Plotting**: We use **Plotly** to create an interactive scatter plot.
    - **Color**: Points are colored by their true cancer type label.
    - **Hover**: Hovering over a point reveals the text snippet and the label. This allows you to inspect specific data points, especially outliers or misclassified examples.
    - _Implementation_: `plot_latent_space_interactive` in `src/interpretability/viz/utils.py`

## 3. Automated Error Analysis

**Goal:** Identify the model's blind spots and understand _why_ it fails on specific examples.

**Methodology:**

1.  **Loss Sorting**: We calculate the **Cross Entropy Loss** for every example in the validation subset.
    - **High Loss**: Indicates the model was very confident in the wrong answer, or very unsure about the right answer. These are the "most confusing" examples.
2.  **Ranking**: We sort examples by loss in descending order.
3.  **Explanation**: For the top confusing examples, we automatically run the Integrated Gradients explainer to visualize which tokens contributed to the prediction (or lack thereof).
    - _Implementation_: `ErrorAnalyzer.find_most_confusing_examples` in `src/interpretability/error_analysis.py`

---

### How to Run

1.  Open `notebooks/global_analysis.ipynb`.
2.  Ensure you have established the `src` module path (handled automatically in the notebook).
3.  Run all cells. The notebook will load the final trained model (`../checkpoints/classifier_run/final_model`) and generate the interactive visualizations.
