import torch 
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenized_chunks = torch.load("Sonnet II.pt")

# Decode each chunk back into text
decoded_texts = []
for chunk in tokenized_chunks:
    # Ensure chunk contains multiple sequences
    for sequence in chunk["input_ids"]:
        # Decode the sequence (list of token IDs) to text
        decoded_text = tokenizer.decode(sequence, skip_special_tokens=True)
        decoded_texts.append(decoded_text)

# Combine all decoded sequences
reconstructed_text = ''.join(decoded_texts)

print(reconstructed_text)