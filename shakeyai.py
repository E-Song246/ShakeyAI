import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
output_dir = "shakespeare_model"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir)

# Ensure model is in evaluation mode
model.eval()

# Define a prompt for text generation
prompt = "Betwixt the ever brancing leaves\n\na simple robin doth deign to build her nest"

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_length=100,  # Maximum length of the generated text
        num_return_sequences=1,  # Number of generated sequences
        temperature=0.7,  # Sampling temperature
        top_p=0.9,  # Nucleus sampling
        top_k=50,  # Top-k sampling
        do_sample=True  # Use sampling for text generation
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
