import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import json

# Load models from other files
from train import run_training_session
from data_loader import get_dataloaders, preprocess_input
from models import BaselineNN, ImprovedNN, BaselineCNN, ImprovedCNN
from param_counter import count_parameters

# Make sure saving directory exists
SAVE_PATH="best_model_weights"
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Experiment arguments ---
class Args:
    def __init__(self, model_name, model_type, epochs=15, batch_size=128, lr=1e-3
                 , val_split=0.07, data_dir='./data',seed=42, device='cuda',
                 # Ablation related args
                use_bn=True, use_dropout=True, dropout_rate=0.25,
                use_residual=True, downsampling_type='stride_conv'
                ):
        self.model_name = model_name
        self.model_type = model_type  # 'NN' or 'CNN'
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr # learning rate
        self.val_split = val_split
        self.data_dir = data_dir
        self.seed = seed
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

        # Ablation related args
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.downsampling_type = downsampling_type

# --- Modeling ablation experiments ---
def get_model_and_type(args: Args) -> tuple:
    """
    Based on args, return the model instance and model type.
    And pass ablation arguments to ImprovedCNN/ImprovedNN if needed.
    """
    if args.model_type=='NN':
        if 'Baseline' in args.model_name:
            # Baseline NN don't have ablation options
            return BaselineNN(), 'NN'
        else:
            # Improved NN with ablation options
            return ImprovedNN(
                dropout_rate=args.dropout_rate,
                use_bn=args.use_bn,
                use_dropout=args.use_dropout
            ), 'NN'
    elif args.model_type=='CNN':
        if 'Baseline' in args.model_name:
            # BaselineCNN don't have ablation options
            return BaselineCNN(), 'CNN'
        else:
            # ImprovedCNN accept all ablation options
            return ImprovedCNN(
                use_bn=args.use_bn,
                use_dropout=args.use_dropout,
                dropout_rate=args.dropout_rate,
                use_residual=args.use_residual,
                downsampling_type=args.downsampling_type
            ), 'CNN'
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
# --- Kaggle submission function ---
def generate_submission(model: nn.Module, model_type: str, args: Args, submission_path: str):
    """
    Use the trained best model, test on kaggle test set, and generate submission CSV file.

    Noted: We must reloaded 'test4students.csv''s DataLoader .
    """
    print(f"\n --- Generating submission for {args.model_name} ({model_type}) ---")

    # 1. Load test dataset
    _, _, test_loader = get_dataloaders(args)

    # 2. Load the best model weights
    model_path=os.path.join(SAVE_PATH, f"{args.model_name}_{args.model_type}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Best model weights not found at {model_path}. Please train the model first.")
        return
    
    # Reload the best model weights
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # 3. Predict on test set
    predictions=[]
    image_ids=[]

    with torch.no_grad():
        for inputs, ids in test_loader:
            # Move to device and preprocess
            inputs=preprocess_input(inputs.to(args.device), model_type)

            outputs=model(inputs)
            # Get the highest probability class
            _, predicted_labels=torch.max(outputs.data,1)

            # Collect predictions and image IDs: In data_loader.py, test_loader's labels is actually image IDs
            predictions.extend(predicted_labels.cpu().numpy())
            image_ids.extend(ids.cpu().numpy())
    # 4. Create submission DataFrame csv
    submission_df=pd.DataFrame({
        'idx': image_ids,
        'label': predictions
    })

    # Save to CSV in submission_path
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

# --- Main ablation experiment function ---
def run_all_experiments():
    """
    Define and run all ablation experiments, save best model weights and generate submissions.
    """      
    # Set base arguments
    BASE_EPOCHS=20
    BASE_LR=1e-3
    BASE_BATCH_SIZE=128

    # --- Define ablation experiments setup ---
    experiments=[
        # NN Baseline and Improved
        Args('NN_Baseline','NN', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        Args('NN_Improved_NoReg','NN', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),

        # NN Ablations Studied
        Args('NN_ablation_NoBN','NN', use_bn=False, use_dropout=True, epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        Args('NN_ablation_NoDropout','NN', use_bn=True, use_dropout=False, epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),

        # CNN Baseline and Improved
        Args('CNN_Baseline','CNN', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        # Improved CNN: +BN, +Dropout, +Residual, +Strided Conv Downsampling
        Args('CNN_Improved_Full','CNN', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),

        # CNN Ablations Studied
        # 1. -BN(keep Dropout, Residual, Strided Conv)
        Args('CNN_ablation_NoBN','CNN', use_bn=False, epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        #2. -Residual (keep BN, Dropout, Use conv+pooling downsampling)
        Args('CNN_ablation_NoResidual','CNN', use_residual=False, epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        #3. -Dropout (keep BN, Residual, Strided Conv)
        Args('CNN_ablation_NoDropout','CNN', use_dropout=False, epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        #4. Pooling: MaxPool downsampling (keep BN, Dropout, Residual)
        Args('CNN_ablation_MaxPool','CNN', downsampling_type='maxpool', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        #5. Pooling: AvgPool downsampling (keep BN, Dropout, Residual)
        Args('CNN_ablation_AvgPool','CNN', downsampling_type='avgpool', epochs=BASE_EPOCHS, lr=BASE_LR, batch_size=BASE_BATCH_SIZE),
        #6. No downsampling (keep BN, Dropout, Residual) -> Expect very high memory usage
        Args('CNN_ablation_NoDownsampling','CNN', downsampling_type='none', epochs=5, lr=BASE_LR, batch_size=BASE_BATCH_SIZE), # Reduce batch size to avoid OOM      
    ]

    # --- Run experiments ---
    all_results={}

    for args in experiments:
        print("="*60)

        #1. Get model and type
        model, model_type=get_model_and_type(args)

        #2. Run training session
        history=run_training_session(
            model=model,
            model_type=model_type,
            args=args
        )

        # 3. Save experiment results
        final_result={
            'Model Name': args.model_name,
            'Model Type': args.model_type,
            'Total Parameters': count_parameters(model),
            'Best Val Acc': history.get('best_val_acc',0.0),
            'History': history # Include Loss and Acc curves
        }

        all_results[args.model_name]=final_result

        # 4. Clear GPU memory
        if args.device=='cuda':
            torch.cuda.empty_cache()

    # --- Save all experiment results to JSON ---
    with open('experiment_results.json','w') as f:
        json.dump(all_results,f,indent=4)
    print("\nAll experiment results saved to experiment_results.json")

    # --- Generate submission files for best models ---
    # 1. Find best model first
    highest_val_acc = 0.0
    best_model_name = None
    
    for name, result in all_results.items():
        if result.get('Best Val Acc', 0.0) > highest_val_acc:
            highest_val_acc = result['Best Val Acc']
            best_model_name = name
    
    if best_model_name is None:
        print("No valid training session completed to select the best model.")
        return
    print('='*60)
    print(f"\nBest model based on validation accuracy: {best_model_name} with Val Acc: {highest_val_acc:.2f}%")
    print('='*60)

    # 2. Generate submission for the best model
    best_args=next(a for a in experiments if a.model_name==best_model_name)

    #3. Load best model instance
    best_model, best_model_type=get_model_and_type(best_args)

    #4. Generate submission CSV
    generate_submission(
        model=best_model,
        model_type=best_model_type,
        args=best_args,
        submission_path=f'submission.csv'
    )

    print("\n All experiments completed. Please check experiment_results.json and submission.csv. --- \n")

if __name__=='__main__':
    # Make sure the training data exists
    if not os.path.exists('./data/train.csv') or not os.path.exists('./data/test4students.csv'):
        print("Error: Missing data files. Please ensure './data/train.csv' and './data/test4students.csv' exist.")
    else:
        run_all_experiments()