from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class ErrorAnalyzer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def find_most_confusing_examples(
        self,
        texts: List[str],
        true_labels: List[int],
        class_names: List[str],
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Identifies examples with the highest loss.
        """
        results = []

        print(f" analyzing {len(texts)} examples for errors...")
        with torch.no_grad():
            for i, (text, label) in enumerate(
                tqdm(zip(texts, true_labels), total=len(texts))
            ):
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Calculate Cross Entropy Loss for this example
                loss = F.cross_entropy(
                    logits, torch.tensor([label]).to(self.device), reduction="none"
                ).item()

                # Get prediction confidence
                probs = F.softmax(logits, dim=1)
                pred_conf, pred_idx = torch.max(probs, dim=1)

                results.append(
                    {
                        "Validation_ID": i,
                        "Text": text,
                        "True_Label": class_names[label],
                        "Predicted_Label": class_names[pred_idx.item()],
                        "Confidence": pred_conf.item(),
                        "Loss": loss,
                        "Correct": (label == pred_idx.item()),
                    }
                )

        df = pd.DataFrame(results)
        # Sort by Loss descending (highest loss = most wrong/confused)
        return df.sort_values("Loss", ascending=False).head(top_k)

    def explain_example(self, row: pd.Series):
        """
        Returns a formatted string explanation for a row from the dataframe.
        """
        status = "✅ Correct" if row["Correct"] else "❌ Incorrect"
        return (
            f"Example ID: {row['Validation_ID']}\n"
            f"Status: {status}\n"
            f"True Label: {row['True_Label']}\n"
            f"Predicted: {row['Predicted_Label']} (Confidence: {row['Confidence']:.4f})\n"
            f"Loss: {row['Loss']:.4f}\n"
            f"Report Excerpt: {row['Text'][:300]}...\n"
        )
