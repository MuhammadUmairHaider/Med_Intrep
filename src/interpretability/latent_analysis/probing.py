import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class ProbingClassifier:
    """
    A linear probe to test if a specific concept (e.g., 'Is Kidney?')
    is linearly separable in the model's activation space.
    """

    def __init__(self, layer_idx: int, concept_name: str):
        self.layer_idx = layer_idx
        self.concept_name = concept_name
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def train(self, activations: np.ndarray, labels: np.ndarray):
        X_train, X_test, y_train, y_val = train_test_split(
            activations, labels, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_test, y_val)

        return val_acc

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.model.predict(activations)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir, f"probe_layer_{self.layer_idx}_{self.concept_name}.joblib"
        )
        joblib.dump(self.model, path)


def run_layer_wise_probing(
    activations_per_layer: Dict[int, np.ndarray], labels: np.ndarray, concept_name: str
) -> Dict[int, float]:
    """
    Trains a probe for every layer and returns the accuracy profile.
    """
    results = {}
    for layer_idx, feats in activations_per_layer.items():
        probe = ProbingClassifier(layer_idx, concept_name)
        acc = probe.train(feats, labels)
        results[layer_idx] = acc
    return results
