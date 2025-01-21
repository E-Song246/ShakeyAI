import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

# Define ShakespeareDataset class
class ShakespeareDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.label_map = {'comedy': 0, 'tragedy': 1, 'history': 2, 'poetry': 3}

        for label_name, label_idx in self.label_map.items():
            folder_path = os.path.join(root_dir, label_name)
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".pt"):
                    file_path = os.path.join(folder_path, file_name)
                    tokenized_chunks = torch.load(file_path)

                    for chunk in tokenized_chunks:
                        self.data.append(chunk)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        label = self.labels[idx]
        input_ids = chunk['input_ids'].squeeze(0)
        attention_mask = chunk['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

# Custom collate function
def custom_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids_padded, attention_masks_padded, labels

# Initialize dataset and dataloader
dataset_path = 'shakespeare_works/tokenized'
shakespeare_dataset = ShakespeareDataset(dataset_path)
data_loader = DataLoader(
    shakespeare_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn
)

# Initialize model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (input_ids, attention_masks, _) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item()}")

# Save the model
output_dir = "shakespeare_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Text generation function
def generate_text(prompt, max_length=100, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example text generation
prompt = "To be or not to be"
print("Generated text:")
print(generate_text(prompt))
