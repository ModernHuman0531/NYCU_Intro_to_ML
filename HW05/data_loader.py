import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# --- Define Dataset Class: Read data frrom CSV and preprocess ---
class FashionMNISTDataset(Dataset):
    def __init__(self,csv_file,mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            mode (string): 'train' or 'test' to specify dataset type.
        """
        # Read the CSV file
        self.data_frame=pd.read_csv(csv_file)
        self.mode=mode

        # Based on csv structure to split the data (Use train.csv's structure as reference)
        # label, pixel1, pixel2, ..., pixel784
        if self.mode=='train' or self.mode=='val':
            self.labels=self.data_frame.iloc[:,0].values
            # Transform pixel to numpy array(unit8, 0-255)
            self.images=self.data_frame.iloc[:,1:].values.astype(np.uint8)

        elif self.mode=='test':
            """
            test csv structure:
            index, label, pixel1, pixel2, ..., pixel784
            Since label is -1 for test data, we do not need to load it.
            """
            self.indices=self.data_frame.iloc[:,0].values
            self.images=self.data_frame.iloc[:,2:].values.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and Reshape to (28, 28)
        image=self.images[idx].reshape(28,28)

        # Convert into Float32 Tensor and Normalize to [-1, 1] to stabilize training
        # (img/255.0-0.5)/0.5 ==> (img-127.5)/127.5
        image=(image.astype(np.float32)-127.5)/127.5

        # Convert dimensions from (28, 28) to (1, 28, 28) for CNN input
        # unseqeeze(0) adds a new dimension at position 0
        image=torch.tensor(image).unsqueeze(0)

        if self.mode=='train' or self.mode=='val':
            label=self.labels[idx]
            # Convert label to tensor, and return img, label
            return image, torch.tensor(label,dtype=torch.long)
        elif self.mode=='test':
            # Don't have to convert index to tensor, since we want to save it as csv directly
            index=self.indices[idx]
            return image, index
def get_dataloaders(args):
    """
    Args:
        args: Include batch_size, val_split, data_dir, seed, etc.
    """
    # Set random seed for reproducibility
    seed=torch.Generator().manual_seed(args.seed)

    # Read Train CSV (train.csv file)
    full_dataset=FashionMNISTDataset(csv_file=args.data_dir+'/train.csv',mode='train')

    # Split dataset into training and validation sets
    val_size=int(len(full_dataset)*args.val_split)
    train_size=len(full_dataset)-val_size
    train_dataset, val_dataset=random_split(
        full_dataset,
        [train_size,val_size],
        generator=seed
    )

    # Read TEST CSV (test4student.csv file)
    test_dataset=FashionMNISTDataset(csv_file=args.data_dir+'/test4students.csv',mode='test')

    # Create DataLoaders for each dataset
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)
    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    print(f"[Data] loaded from csv files. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def preprocess_input(x: torch.Tensor, model_type: str) -> torch.Tensor:
    """
    Preprocess input tensor based on model type.
    Args:
        x (torch.Tensor): Input tensor.
        model_type (str): Type of model ('NN' or 'CNN').
    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    if model_type == 'NN':
        # NN(MLP): flatten each 28*28 to a 784-dim vector
        # x.shape changes from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(x.size(0), -1)  # (batch_size, 784)
    elif model_type == 'CNN':
        # Ensure input is in (batch_size, 1, 28, 28) format for CNN
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
    return x
