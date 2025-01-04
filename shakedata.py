import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class ShakespeareDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialize the dataset.
        Args:
            root_dir (str): Path to the root directory containing tokenized subfolders.
        """
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
                        # Append the actual data and label
                        self.data.append(chunk)
                        self.labels.append(label_idx)

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
        chunk = self.data[idx]
        label = self.labels[idx]
        
        # Extract input_ids and attention_mask
        input_ids = chunk['input_ids'].squeeze(0)  # Remove batch dimension if present
        attention_mask = chunk['attention_mask'].squeeze(0)  # Remove batch dimension if present
        
        return input_ids, attention_mask, label

# Path to the 'tokenized' directory
dataset_path = 'shakespeare_works/tokenized'

# Instantiate the dataset
shakespeare_dataset = ShakespeareDataset(dataset_path)

def custom_collate_fn(batch):
    """
    Custom collate function to pad input_ids and attention masks.
    Args:
        batch (list): List of tuples (input_ids, attention_mask, label).
    Returns:
        Padded input_ids, attention_masks, and labels.
    """
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return input_ids_padded, attention_masks_padded, labels

# Create a DataLoader with the custom collate function
data_loader = DataLoader(
    shakespeare_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn
)

# Example: Iterate over the DataLoader
for batch in data_loader:
    input_ids, attention_masks, labels = batch
    print("Input IDs shape:", input_ids.size())
    print("Attention Masks shape:", attention_masks.size())
    print("Labels:", labels)
    break
