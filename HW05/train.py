import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import sys

# Load data loader and models
from data_loader import get_dataloaders, preprocess_input
from models import BaselineNN, ImprovedNN, BaselineCNN, ImprovedCNN
from param_counter import count_parameters

NUM_CLASSES=10  # Number of classes for FashionMNIST
# --- Define Args class to hold hyperparameters and settings ---
class Args:
    def __init__(self, model_name, epochs, batch_size, lr, val_split, data_dir, seed, device='cuda'):
            self.model_name = model_name
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr # learning rate
            self.val_split = val_split
            self.data_dir = data_dir
            self.seed = seed
            self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

# --- Set environment and seed in order to ensure reproducibility ---
def set_seed(seed: int):
    """
    Fixed random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed} on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- Validation function ---
def validate_model(model: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, model_type: str, device: str) -> tuple:
    """
    Use validation dataset to evaluate the model performance.
    Use tqdm to show progress bar.
    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation dataset.
        criterion (nn.Module): Loss function.
        model_type (str): Type of model ('NN' or 'CNN').
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    model.eval()  # Set model to evaluation mode
    total_loss=0.0
    correct_predictions=0
    total_samples=0

    # use tqdm to show prtogress bar
    val_tqdm=tqdm(val_loader, desc="Validating", leave=False, file=sys.stdout)

    # Disable gradient calculation for validation
    with torch.no_grad():
        for inputs, labels in val_tqdm:
            # Move data to device and preprocess
            inputs=preprocess_input(inputs.to(device), model_type)
            labels=labels.to(device)

            outputs=model(inputs)
            loss=criterion(outputs, labels)

            total_loss+=loss.item()*inputs.size(0)

            # Calculate accuracy
            _, predicted=torch.max(outputs,1)
            correct_predictions+=(predicted==labels).sum().item()
            total_samples+=labels.size(0)

            # Update tqdm description
            val_tqdm.set_postfix(Loss=loss.item())
    avg_loss=total_loss/total_samples
    accuracy=correct_predictions/total_samples * 100.0

    return avg_loss, accuracy

# --- Training function one epoch ---
def train_one_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, model_type: str, device: str) -> tuple:
    """
    Train the model for one epoch.
    Use tqdm to show progress bar.
    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training dataset.
        optimizer (Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        model_type (str): Type of model ('NN' or 'CNN').
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    model.train()  # Set model to training mode
    total_loss=0.0
    correct_predictions=0
    total_samples=0

    # use tqdm to show progress bar
    train_tqdm=tqdm(train_loader, desc="Training", leave=False, file=sys.stdout)

    for batch_idx, (inputs, labels) in enumerate(train_tqdm):
        # Move data to device and preprocess
        inputs=preprocess_input(inputs.to(device),model_type)
        labels=labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs=model(inputs)

        # Compute loss
        loss=criterion(outputs, labels)

        # Backward pass and optimization the model parameters
        loss.backward()
        optimizer.step()

        # Calculate the statistics
        total_loss+=loss.item()*inputs.size(0)

        # Calculate accuracy per batch
        _, predicted=torch.max(outputs,1) # Choose the class with highest probability
        correct_predictions+=(predicted==labels).sum().item()
        total_samples+=labels.size(0)

        # Update tqdm description
        train_tqdm.set_postfix(Loss=loss.item(), Acc=f"{correct_predictions/total_samples*100.0:.2f}%")

    avg_loss=total_loss/total_samples
    accuracy=correct_predictions/total_samples * 100.0

    return avg_loss, accuracy

# --- Main training function (run_training_session)---
def run_training_session(model: nn.Module, model_type: str, args: Args, **kwargs):
    """
    Main training function to run the training session.
    Args:
        model (nn.Module): The neural network model.
        model_type (str): Type of model ('NN' or 'CNN').
        args (Args): Hyperparameters and settings.
        **kwargs: Additional keyword arguments.
    """
    print(f"\n===== Starting Training Session for {args.model_name} ({model_type})=====")

    # Set random seed
    set_seed(args.seed)

    # Load data loaders
    try:
        train_loader, val_loader, test_loader=get_dataloaders(args)
    
    except Exception as e:
        print(f"Error loading data: {e}. Check data_loader.py and data directory.")
        return {}
    
    # Initialize model, criterion, optimizer
    model=model.to(args.device)
    criterion=nn.CrossEntropyLoss() # For multi-class classification use CrossEntropyLoss
    optimizer=optim.Adam(model.parameters(), lr=args.lr)

    # Record training history
    history={
        'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'best_val_acc':0.0
    }

    # Print the number of parameters
    total_params=count_parameters(model)
    print(f"Model has {total_params} trainable parameters.")

    # Training loop
    for epoch in range(1, args.epochs+1):
        start_time=time.time()

        # Train for one epoch(train_one_epoch will show progress bar)
        train_loss, train_acc=train_one_epoch(model, train_loader, optimizer, criterion, model_type, args.device)

        # Validate the model(validate_model will show progress bar)
        val_loss, val_acc=validate_model(model, val_loader, criterion, model_type, args.device)
        end_time=time.time()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n[Epoch {epoch}/{args.epochs}] | Time: {end_time-start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc']=val_acc
            model_path=f"best_model_weights/{args.model_name}_{model_type}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Save model weights
            torch.save(model.state_dict(), model_path)
            print(f"-> Model saved to {model_path} with best Val Acc: {val_acc:.2f}%")

    print(f"\n===== Training finish. Best Validation Accuracy: {history['best_val_acc']:.2f}% =====")

    return history