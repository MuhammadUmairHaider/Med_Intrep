from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpretability.feature_importance.integrated_gradients_classifier import \
    ClassifierIG


class GlobalExplainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cig = ClassifierIG(model, tokenizer)

    def accumulate_token_importance(
        self,
        texts: List[str],
        target_indices: List[int],
        n_steps: int = 20,
        batch_size: int = 8,
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Accumulates token importance scores across a dataset.
        Returns:
            Dict[class_idx, Dict[token, List[score]]]
        """
        global_scores = defaultdict(lambda: defaultdict(list))

        # Process in batches? Actually IG is slow, maybe one by one with progress bar
        print(f"Calculating global importance for {len(texts)} examples...")

        for i, (text, target_idx) in enumerate(
            tqdm(zip(texts, target_indices), total=len(texts))
        ):
            try:
                # Get local importance
                res = self.cig.interpret(
                    text, target_class_idx=target_idx, n_steps=n_steps
                )

                # Accumulate
                for token, score in zip(res["tokens"], res["scores"]):
                    # Clean token
                    clean_token = token.replace("Ġ", "").replace(" ", "").strip()
                    if not clean_token or clean_token in [
                        "<bos>",
                        "<eos>",
                        "<pad>",
                        "<unk>",
                    ]:
                        continue

                    # Store score for this class
                    global_scores[target_idx][clean_token].append(score)

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        return global_scores

    def get_top_global_tokens(
        self,
        global_scores: Dict[int, Dict[str, List[float]]],
        class_idx: int,
        k: int = 20,
    ) -> pd.DataFrame:
        """
        Aggregates scores (mean) and returns top-k tokens for a class.
        Using mean score highlights tokens that consistently push towards the class.
        """
        if class_idx not in global_scores:
            return pd.DataFrame()

        token_data = []
        for token, scores in global_scores[class_idx].items():
            scores_arr = np.array(scores)
            mean_score = np.mean(scores_arr)
            # We care about magnitude but direction matters. Positive mean = generally indicative of class.
            # Negative mean = generally indicative against class.

            token_data.append(
                {
                    "Token": token,
                    "MeanScore": mean_score,
                    "Count": len(scores),
                    "AbsMeanScore": abs(mean_score),
                }
            )

        df = pd.DataFrame(token_data)
        if df.empty:
            return df

        # Filter rare tokens? e.g. appear at least 3 times
        df = df[df["Count"] >= 3]

        return df.sort_values("AbsMeanScore", ascending=False).head(k)
