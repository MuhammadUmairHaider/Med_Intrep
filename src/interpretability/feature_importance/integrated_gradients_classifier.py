from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizer


class ClassifierIG:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        # Ensure model is in eval mode
        self.model.eval()

    def _forward_func(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        target_class_idx: int = 0,
    ):
        """
        Custom forward function for Captum for Sequence Classification.
        Returns the logit for the target class.
        """
        if (
            attention_mask is not None
            and attention_mask.shape[0] != inputs_embeds.shape[0]
        ):
            attention_mask = attention_mask.repeat(inputs_embeds.shape[0], 1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # Output logits shape: [batch_size, num_labels]
        logits = outputs.logits

        # Return the logit for the target class
        return logits[:, target_class_idx]

    def interpret(
        self,
        input_text: str,
        target_class_idx: int,
        n_steps: int = 50,
        internal_batch_size: int = None,
    ) -> Dict[str, Any]:

        # Ensure target_class_idx is a standard python int, as numpy ints can cause issues with Captum
        target_class_idx = int(target_class_idx)

        encoded = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        ig = IntegratedGradients(self._forward_func)

        baseline_embeds = torch.zeros_like(inputs_embeds)

        if internal_batch_size is None:
            internal_batch_size = 4

        # Attribute
        # target=target_class_idx tells Captum which output neuron to attribute to
        # But since our _forward_func takes target_class_idx as an arg to select the slice,
        # we pass it via additional_forward_args.
        # Wait, IntegratedGradients `attribute` target param creates a slice
        # IF the forward function returns a tensor of outputs.
        # BUT our forward function returns a single scalar (logit) per batch item if we handle the index inside.
        # Let's align with the standard pattern:
        # Forward returns [batch_size, num_classes], target points to the column.

        # Re-defining forward to return full logits
        def forward_wrapper(embeds, mask):
            outputs = self.model(inputs_embeds=embeds, attention_mask=mask)
            return outputs.logits

        ig = IntegratedGradients(forward_wrapper)

        attributions, delta = ig.attribute(
            inputs=inputs_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_class_idx,
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
            "target_class_idx": target_class_idx,
            "delta": delta.item(),
        }
