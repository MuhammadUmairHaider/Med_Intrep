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
word_counter = len(word_spans)  # Start new IDs after words

for start, end in offsets:
    if start == end:
        token_to_word.append(
            word_counter
        )  # Assign special token to a new unmapped unique gate
        word_counter += 1
    else:
        # Find word with max overlap
        best_word = -1
        max_overlap = -1
        for w_idx, (w_start, w_end) in enumerate(word_spans):
            overlap = max(0, min(end, w_end) - max(start, w_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_word = w_idx
        if best_word == -1:
            # Just in case token maps to nothing (e.g. pure whitespace)
            token_to_word.append(word_counter)
            word_counter += 1
        else:
            token_to_word.append(best_word)

print("Tokens:", tokens)
print("Token -> Word ID:", token_to_word)
print("Total unique gates:", word_counter)

for i in range(len(tokens)):
    w_id = token_to_word[i]
    if w_id < len(word_spans):
        w_start, w_end = word_spans[w_id]
        print(f"Token '{tokens[i]}' -> Word '{text[w_start:w_end]}'")
    else:
        print(f"Token '{tokens[i]}' -> Extranous/Special Token")
