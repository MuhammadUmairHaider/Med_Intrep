import os
import re
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# Add src to path if needed to import l0
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from l0 import HardConcreteGate


class AdversarialPatchingExplainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        # Ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def interpret(
        self,
        input_text: str,
        target_class_idx: int = None,
        kl_threshold: float = 0.05,
        max_epochs: int = 500,
        lr: float = 0.1,
        lr_lambda: float = 0.01,
        initial_lambda: float = 1.0,
        sparsity_weight: float = 1.0,
        completeness_weight: float = 0.0,
        baseline_type: str = "unk",
        use_task_loss: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimizes an L0 gate over the input text tokens to find the sparsest
        mask that keeps the output KL divergence below kl_threshold.
        """
        # 1. Tokenize Input
        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        seq_len = input_ids.shape[1]

        # 1.5 Map subword tokens to chunks using delimiter boundaries
        offsets = encoded["offset_mapping"][0].cpu().numpy()
        word_spans = [
            m.span() for m in re.finditer(r"[^.,\-:]+[.,\-:]*|[.,\-:]+", input_text)
        ]

        token_to_word = []
        word_counter = len(word_spans)

        for start, end in offsets:
            if start == end:
                token_to_word.append(word_counter)
                word_counter += 1
            else:
                best_word = -1
                max_overlap = -1
                for w_idx, (w_start, w_end) in enumerate(word_spans):
                    # Calculate overlap between token boundaries and regex word boundaries
                    overlap = max(0, min(end, w_end) - max(start, w_start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_word = w_idx
                if best_word == -1:
                    token_to_word.append(word_counter)
                    word_counter += 1
                else:
                    token_to_word.append(best_word)

        num_gates = word_counter
        # Create a boolean/float matrix mapping tokens to words: shape [seq_len, num_gates]
        # M_token = matrix @ M_words
        token_to_word_matrix = torch.zeros(
            (seq_len, num_gates), device=self.device, dtype=torch.bfloat16
        )  # Use bfloat16 to match the model embeddings
        for i, w_idx in enumerate(token_to_word):
            token_to_word_matrix[i, w_idx] = 1.0

        # 2. Get Original Forward Pass Context
        embedding_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            E_orig = embedding_layer(input_ids)  # [1, seq_len, hidden_dim]
            outputs_orig = self.model(
                inputs_embeds=E_orig, attention_mask=attention_mask
            )
            logits_orig = outputs_orig.logits.detach()  # [1, num_classes]

            if target_class_idx is None:
                target_class_idx = logits_orig.argmax().item()
            target_tensor = torch.tensor([target_class_idx], device=self.device)

            # 3. Create baseline embedding based on baseline_type
            if baseline_type == "unk":
                base_token_id = self.tokenizer.unk_token_id
                if base_token_id is None:
                    base_token_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else 0
                    )
                base_id_tensor = torch.tensor([[base_token_id]], device=self.device)
                E_base = embedding_layer(base_id_tensor)  # [1, 1, hidden_dim]
            elif baseline_type == "pad":
                base_token_id = self.tokenizer.pad_token_id
                if base_token_id is None:
                    base_token_id = 0
                base_id_tensor = torch.tensor([[base_token_id]], device=self.device)
                E_base = embedding_layer(base_id_tensor)
            elif baseline_type == "zero":
                E_base = torch.zeros(
                    (1, 1, E_orig.shape[-1]), device=self.device, dtype=E_orig.dtype
                )
            else:
                raise ValueError(f"Unknown baseline_type: {baseline_type}")

            # If E_base is exactly zero (e.g., zero baseline or pad token in some models), add small noise so RMSNorm doesn't blow up
            if E_base.abs().sum() < 1e-5:
                # Add small normal noise
                E_base = torch.randn_like(E_base) * 0.01

            E_base = E_base.expand(1, seq_len, -1)  # [1, seq_len, hidden_dim]

        # 4. Initialize L0 gates
        # One gate logit for each unique word (not subword token)
        gate = HardConcreteGate(size=num_gates).to(self.device)

        optimizer = torch.optim.AdamW(gate.parameters(), lr=lr)

        # Lagrangian multiplier for the constraint
        lambda_reg = initial_lambda

        best_mask = None
        best_gate_values = None
        best_sparsity = 0.0
        best_kl = float("inf")

        pbar = tqdm(range(max_epochs), desc="Optimizing gates", unit="epoch")
        for epoch in pbar:
            gate.train()
            optimizer.zero_grad()

            # Forward pass through gate
            M_words = gate()  # [num_gates]
            M = token_to_word_matrix @ M_words.to(
                token_to_word_matrix.dtype
            )  # [seq_len] map word gates to token mask
            M_expanded = M.view(1, seq_len, 1)  # [1, seq_len, 1]

            # Patch embeddings: M=1 keeps orig, M=0 uses baseline
            # Ensure mask matches dtype of embeddings (e.g. float16)
            M_expanded = M_expanded.to(E_orig.dtype)
            E_patched = M_expanded * E_orig + (1.0 - M_expanded) * E_base

            # Forward pass through model
            outputs_patched = self.model(
                inputs_embeds=E_patched, attention_mask=attention_mask
            )
            logits_patched = outputs_patched.logits

            # KL Divergence: KL(original || patched)
            # Add eps for numerical stability
            if torch.isnan(logits_patched).any():
                print(f"Epoch {epoch}: NaN detected in logits_patched!")
                if torch.isnan(E_patched).any():
                    print(f"Epoch {epoch}: NaN detected in E_patched!")
                if torch.isnan(M).any():
                    print(f"Epoch {epoch}: NaN detected in M!")

            log_probs_patched = F.log_softmax(logits_patched.float(), dim=-1)
            probs_orig = F.softmax(logits_orig.float(), dim=-1)

            # Using exact formulation for KL
            loss_task = F.kl_div(log_probs_patched, probs_orig, reduction="batchmean")

            if use_task_loss:
                # Add cross entropy loss to guide patching towards the target class
                ce_loss = F.cross_entropy(logits_patched.float(), target_tensor)
                loss_task = loss_task + ce_loss

            if torch.isnan(loss_task):
                print(f"Epoch {epoch}: NaN detected in loss_task!")
                loss_task = torch.tensor(10.0, device=self.device, requires_grad=True)

            # Completeness Loss on unselected tokens mapping
            loss_completeness = torch.tensor(0.0, device=self.device)
            if completeness_weight > 0.0:
                M_complement = 1.0 - M_expanded
                E_complement = M_complement * E_orig + (1.0 - M_complement) * E_base
                outputs_complement = self.model(
                    inputs_embeds=E_complement, attention_mask=attention_mask
                )
                logits_complement = outputs_complement.logits
                probs_uniform = (
                    torch.ones_like(logits_complement) / logits_complement.shape[-1]
                )
                loss_completeness = F.cross_entropy(
                    logits_complement.float(), probs_uniform
                )

            # Sparsity Loss
            loss_sparsity = gate.get_sparsity_loss()

            # Total Loss (Lagrangian formulation)
            # Minimize: sparsity + lambda * (task_loss - threshold) + completeness
            # Therefore: L = sparsity_weight * loss_sparsity + lambda_reg * loss_task + completeness_weight * loss_completeness
            loss = (
                (sparsity_weight * loss_sparsity)
                + lambda_reg * loss_task
                + (completeness_weight * loss_completeness)
            )

            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)

            optimizer.step()

            # Ensure gates don't explode (hard limits to keep sigmoid in safe ranges)
            with torch.no_grad():
                gate.log_alpha.clamp_(-10.0, 10.0)

            # Update lambda (Lagrangian multiplier update for constraint: loss_task <= kl_threshold)
            # Since lambda_reg is now multiplying loss_task:
            # If loss_task > threshold, lambda_reg INCREASES (to focus more on reducing task loss).
            # If loss_task < threshold, lambda_reg DECREASES (to focus more on sparsity).
            with torch.no_grad():
                lambda_reg += lr_lambda * (loss_task.item() - kl_threshold)
                lambda_reg = max(0.0, lambda_reg)  # Keep lambda positive

                # Check metrics in deterministic mode
                gate.eval()
                eval_M_words = gate()
                eval_M = token_to_word_matrix @ eval_M_words.to(
                    token_to_word_matrix.dtype
                )
                eval_M_expanded = eval_M.view(1, seq_len, 1).to(E_orig.dtype)
                eval_E_patched = (
                    eval_M_expanded * E_orig + (1.0 - eval_M_expanded) * E_base
                )
                eval_logits = self.model(
                    inputs_embeds=eval_E_patched, attention_mask=attention_mask
                ).logits

                eval_log_probs = F.log_softmax(eval_logits.float(), dim=-1)
                eval_loss_task = F.kl_div(
                    eval_log_probs, probs_orig.float(), reduction="batchmean"
                ).item()
                if use_task_loss:
                    eval_ce_loss = F.cross_entropy(
                        eval_logits.float(), target_tensor
                    ).item()
                    eval_loss_task += eval_ce_loss

                eval_sparsity_rate = gate.get_sparsity_rate()  # fraction of zeros

                eval_loss_completeness = 0.0
                if completeness_weight > 0.0:
                    eval_M_comp = 1.0 - eval_M_expanded
                    eval_E_comp = eval_M_comp * E_orig + (1.0 - eval_M_comp) * E_base
                    eval_logits_comp = self.model(
                        inputs_embeds=eval_E_comp, attention_mask=attention_mask
                    ).logits
                    probs_uniform = (
                        torch.ones_like(eval_logits_comp) / eval_logits_comp.shape[-1]
                    )
                    eval_loss_completeness = F.cross_entropy(
                        eval_logits_comp.float(), probs_uniform
                    ).item()

                # Update progress bar with current metrics
                postfix_dict = {
                    "KL": f"{eval_loss_task:.4f}",
                    "sparsity": f"{eval_sparsity_rate*100:.1f}%",
                    "λ": f"{lambda_reg:.3f}",
                    "loss": f"{loss.item():.4f}",
                }
                if completeness_weight > 0.0:
                    postfix_dict["comp"] = f"{eval_loss_completeness:.4f}"
                pbar.set_postfix(postfix_dict)

                is_correct = eval_logits.argmax().item() == target_tensor.item()
                if is_correct and eval_sparsity_rate > best_sparsity:
                    best_sparsity = eval_sparsity_rate
                    best_mask = eval_M.detach().float().cpu().numpy()

                    # Convert continuous word gate to continuous token gate
                    best_gate_words = gate.get_continuous_mask()
                    best_gate_tokens = token_to_word_matrix @ best_gate_words.to(
                        token_to_word_matrix.dtype
                    )
                    best_gate_values = best_gate_tokens.detach().float().cpu().numpy()

                    best_kl = eval_loss_task
                    best_patched_class_idx = eval_logits.argmax().item()

        if best_mask is None:
            gate.eval()
            eval_M_words = gate()
            eval_M = token_to_word_matrix @ eval_M_words.to(token_to_word_matrix.dtype)
            best_mask = eval_M.detach().float().cpu().numpy()

            best_gate_words = gate.get_continuous_mask()
            best_gate_tokens = token_to_word_matrix @ best_gate_words.to(
                token_to_word_matrix.dtype
            )
            best_gate_values = best_gate_tokens.detach().float().cpu().numpy()

            # Predict with latest mask
            eval_M_expanded = eval_M.view(1, seq_len, 1).to(E_orig.dtype)
            eval_E_patched = eval_M_expanded * E_orig + (1.0 - eval_M_expanded) * E_base
            eval_logits = self.model(
                inputs_embeds=eval_E_patched, attention_mask=attention_mask
            ).logits

            final_kl = loss_task.item() if max_epochs > 0 else 0.0
            best_patched_class_idx = eval_logits.argmax().item()
        else:
            final_kl = best_kl

        # Get tokens
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # We return scores such that 1.0 means kept (important), 0.0 means patched (unimportant)
        return {
            "tokens": input_tokens,
            "scores": best_mask.squeeze(),
            "gate_values": best_gate_values.squeeze(),
            "kl_divergence": final_kl,
            "sparsity_rate": best_sparsity,
            "target_class_idx": (
                target_class_idx
                if target_class_idx is not None
                else logits_orig.argmax().item()
            ),
            "patched_class_idx": best_patched_class_idx,
        }
