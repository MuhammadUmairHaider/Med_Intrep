import os
import sys
import unittest

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Need to add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
# Allow importing from current directory if running from src
sys.path.append(os.getcwd())

from interpretability.feature_importance.global_importance import \
    GlobalExplainer  # noqa: E402


class TestGlobalExplainer(unittest.TestCase):
    def setUp(self):
        # Use a tiny model for fast testing
        model_name = "prajjwal1/bert-tiny"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.explainer = GlobalExplainer(self.model, self.tokenizer)

    def test_accumulate_token_importance(self):
        texts = ["This is a test sentence.", "This is another example."]
        target_indices = [0, 1]

        global_scores = self.explainer.accumulate_token_importance(
            texts, target_indices, n_steps=5
        )

        # Check if we got something back
        self.assertTrue(len(global_scores) > 0)
        # Check if we have scores for class 0 and 1
        self.assertIn(0, global_scores)
        self.assertIn(1, global_scores)

        # Check if tokens are cleaned and present
        # "test" should be in class 0 text
        self.assertTrue("test" in global_scores[0])
        self.assertTrue(len(global_scores[0]["test"]) > 0)

    def test_get_top_global_tokens(self):
        # Mock some global scores data
        # Class 0: "feature1" -> [0.5, 0.4], "feature2" -> [-0.1]
        mock_scores = {
            0: {
                "feature1": [0.5, 0.4, 0.6],  # consistently positive
                "feature2": [-0.1, -0.2, -0.05],  # consistently negative
                "rare": [0.9],  # rare but high
            }
        }

        df = self.explainer.get_top_global_tokens(mock_scores, 0, k=5)

        # Should return dataframe
        self.assertFalse(df.empty)
        # Should contain "feature1" and "feature2"
        self.assertTrue("feature1" in df["Token"].values)
        self.assertTrue("feature2" in df["Token"].values)

        # "rare" should be filtered out because count < 3 (hardcoded in get_top_global_tokens)
        self.assertFalse("rare" in df["Token"].values)


if __name__ == "__main__":
    unittest.main()
