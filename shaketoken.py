import os
import torch
from transformers import GPT2Tokenizer

def tokenize_and_save(input_folder, output_folder, tokenizer, max_length=1024):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".txt", ".pt"))
        
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into chunks of `max_length`
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        tokenized_chunks = []

        for chunk in chunks:
            # Tokenize each chunk
            tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest")
            tokenized_chunks.append(tokens)
        
        # Save tokenized data
        torch.save(tokenized_chunks, output_path)


# Paths
base_path = "shakespeare_works"
categories = ["comedy", "tragedy", "history", "poetry"]

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add special tokens
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<COMEDY>", "<TRAGEDY>", "<HISTORY>", "<POEM>"],
    "pad_token": "<|endoftext|>",  # Use <|endoftext|> as the padding token
})

# Tokenize each category
for category in categories:
    input_folder = os.path.join(base_path, category)
    output_folder = os.path.join(base_path, "tokenized", category)
    tokenize_and_save(input_folder, output_folder, tokenizer)

print("Tokenization complete!")
