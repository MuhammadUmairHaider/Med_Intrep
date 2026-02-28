import re

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("google/medgemma-1.5-4b-it")
text = "The patient presents with ductal carcinoma."
res = t(text, return_offsets_mapping=True)

offsets = res["offset_mapping"]
tokens = t.convert_ids_to_tokens(res["input_ids"])

# 1. Find word spans
word_spans = [m.span() for m in re.finditer(r"[a-zA-Z0-9]+|[^\w\s]", text)]
print("Word Spans:", word_spans)

# 2. Map tokens to words
token_to_word = []
num_words = len(word_spans)

for start, end in offsets:
    if start == end:
        token_to_word.append(
            num_words
        )  # Assign special token to a new unmapped unique gate
        num_words += 1
    else:
        # Find word with max overlap
        best_word = -1
        max_overlap = -1
        for w_idx, (w_start, w_end) in enumerate(word_spans):
            overlap = max(0, min(end, w_end) - max(start, w_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_word = w_idx
        token_to_word.append(best_word)

print("Tokens:", tokens)
print("Token -> Word:", token_to_word)

for i in range(len(tokens)):
    word_idx = token_to_word[i]
    if word_idx < len(word_spans):
        w_start, w_end = word_spans[word_idx]
        print(f"Token '{tokens[i]}' -> Word '{text[w_start:w_end]}'")
    else:
        print(f"Token '{tokens[i]}' -> Special Token")
