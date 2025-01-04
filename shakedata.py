import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ShakespeareDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialize the dataset.
        Args:
            root_dir (str): Path to the root directory containing tokenized subfolders.
        """
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        
        self.label_map = {'comedy': 0, 'tragedy': 1, 'history': 2, 'poetry': 3}
        for label, folder in self.label_map.items():
            folder_path = os.path.join(root_dir, label)
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".pt"):
                    file_path = os.path.join(folder_path, file_name)
                    # Load the file to count the chunks
                    tokenized_chunks = torch.load(file_path)
                    for chunk in tokenized_chunks:
                        self.data.append((file_path, chunk))  # Save file path and chunk reference
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.
        Args:
            idx (int): Index of the item to fetch.
        Returns:
            Tuple[Tensor, Tensor, int]: Input IDs, attention mask, and corresponding label.
        """
        file_path, chunk_ref = self.data[idx]
        label = self.label_map[self.labels[idx]]
        
        tokenized_chunks = torch.load(file_path)  # Load the chunks from file
        chunk = tokenized_chunks[chunk_ref]  # Access the specific chunk
        input_ids = chunk['input_ids']
        attention_mask = chunk['attention_mask']
        
        return input_ids, attention_mask, label

# Path to the 'tokenized' directory
dataset_path = 'tokenized'

# Instantiate the dataset
shakespeare_dataset = ShakespeareDataset(dataset_path)

# Create a DataLoader
data_loader = DataLoader(shakespeare_dataset, batch_size=16, shuffle=True)

# Example: Iterate over the DataLoader
for batch in data_loader:
    input_ids, attention_masks, labels = batch
    print("Input IDs shape:", input_ids.size())
    print("Attention Masks shape:", attention_masks.size())
    print("Labels:", labels)
    break
