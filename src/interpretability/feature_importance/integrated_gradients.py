from typing import Any, Dict

import torch
from captum.attr import IntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizer


class LLMIsDefault:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        # We need to wrap the model to get the output logits for a specific target token
        self.model.eval()

    def _forward_func(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        target_token_index: int = -1,
    ):
        """
        Custom forward function for Captum.
        """
        if (
            attention_mask is not None
            and attention_mask.shape[0] != inputs_embeds.shape[0]
        ):
            attention_mask = attention_mask.repeat(inputs_embeds.shape[0], 1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        if target_token_index == -1:
            return logits[:, -1, :]
        else:
            return logits[:, target_token_index, :]

    def interpret(
        self,
        input_text: str,
        target_label_idx: int = None,
        n_steps: int = 50,
        internal_batch_size: int = None,
    ) -> Dict[str, Any]:

        encoded = self.tokenizer(input_text, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        ig = IntegratedGradients(self._forward_func)

        if target_label_idx is None:
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask
                )
                target_label_idx = outputs.logits[0, -1, :].argmax().item()

        baseline_embeds = torch.zeros_like(inputs_embeds)

        if internal_batch_size is None:
            internal_batch_size = 4

        attributions, delta = ig.attribute(
            inputs=inputs_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask, -1),
            target=target_label_idx,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )

        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        attributions_norm = attributions_sum / torch.norm(attributions_sum)

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "tokens": input_tokens,
            "scores": attributions_norm.float().detach().cpu().numpy(),
            "target_token": self.tokenizer.decode([target_label_idx]),
            "delta": delta.item(),
        }
