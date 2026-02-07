from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_tsne(
    embeddings: np.ndarray, n_components: int = 2, perplexity: int = 30
) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def compute_pca(
    embeddings: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(embeddings)
    return transformed, pca.explained_variance_ratio_
