import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/medgemma-1.5-4b-it"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

text = "Test input for debugging."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("1. Regular forward pass...")
with torch.no_grad():
    out1 = model(**inputs)
    print("Success.")

print("2. Get embeddings...")
embed_layer = model.get_input_embeddings()
inputs_embeds = embed_layer(inputs.input_ids)
print(f"Embeddings shape: {inputs_embeds.shape}")

print("3. Forward pass with inputs_embeds...")
try:
    # We must pass attention_mask as well usually
    out2 = model(inputs_embeds=inputs_embeds, attention_mask=inputs.attention_mask)
    print("Success with inputs_embeds.")
except Exception as e:
    print(f"Failed: {e}")
