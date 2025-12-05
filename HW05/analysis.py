import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Result of evaluation stored in experiment_results.json
RESULTS_FILE='experiment_results.json'

# 1. Load results from JSON file 
def load_results(file_path: str) -> dict:
    """ Load the file that stores experiment results in JSON format. """
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at {file_path}. Please run evaluation first.")
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return None
    

# 2. Generate report table(different models and their acc)
def generate_summary_table(all_results: dict):
    """
    Generate the table that report need: model name, parameter amount, best val acc.
    Use for Design and Parameter Accounting and Experimental Rigor part
    """
    data_for_table=[]

    # Select data from all_results dict
    for model_name, result in all_results.items():

        formatted_params=f"{result['Total Parameters']:,}"  # Format with commas

        data_for_table.append({
            'Model Name': model_name,
            'Model Type': result['Model Type'],
            'Total Parameters': formatted_params,
            'Best Val Acc (%)': f"{result['Best Val Acc']:.2f}"
        })

    df_report=pd.DataFrame(data_for_table)

    print("\n"+"="*50)
    print("Experiment Summary Table:")
    print('='*50)
    latex_code=df_report.to_latex(
        index=False,
        float_format="%.2f",
        caption="Ablation Study Results Summary",
        label="tab:ablation_summary"
    )
    print("\n"+"="*50)
    print(latex_code)
    
    # Save to CSV
    df_report.to_csv('report_summary.csv', index=False)
    print("\nSummary table saved to report_summary.csv")

    return df_report

# 3. Visualize learning curves
def plot_learning_curves(model_name: str, history: dict, output_dir: str='plots'):
    """
    Plot training and validation loss/accuracy curves.
    Args:
        model_name (str): Name of the model.
        history (dict): Training history containing loss and accuracy.
        output_dir (str): Directory to save plots.
    """
    if not history:
        print(f"No history data found for {model_name}")
        return

    epoches=range(1, len(history['train_loss'])+1)

    # Set output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss Curves
    plt.figure(figsize=(6,5))
    plt.plot(epoches, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epoches, history['val_loss'], label='Val Loss', color='orange')
    plt.title(f"{model_name} - Loss Curves")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    loss_path=os.path.join(output_dir, f"{model_name}_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    # Plot Accuracy Curves
    plt.figure(figsize=(6,5))
    plt.plot(epoches, history['train_acc'], label='Train Acc', color='green')
    plt.plot(epoches, history['val_acc'], label='Val Acc', color='red')
    plt.title(f"{model_name} - Accuracy Curves")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    acc_path=os.path.join(output_dir, f"{model_name}_acc_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Curves for {model_name} saved to {output_dir}/")

# 4. Visualize: Ablation Comparison Bar Chart
def plot_ablation_comparison(df_summary: pd.DataFrame, model_type: str, output_dir: str='plots'):
    """
    Draw bar chart to compare different ablation study results.
    Use for Analysis and Insight part.
    """

    # Compared model based on model_type
    df_filtered=df_summary[df_summary['Model Type']==model_type].copy()

    if df_filtered.empty:
        print(f"No data found for model type: {model_type}")
        return
    
    # Convert 'Best Val Acc (%)' to float for sorting
    df_filtered['Best Val Acc (%)']=df_filtered['Best Val Acc (%)'].str.replace('%','').astype(float)

    # Set bar chart parameters
    plt.figure(figsize=(10,6))
    bars=plt.bar(df_filtered['Model Name'], df_filtered['Best Val Acc (%)'], color='skyblue')

    # Highlighting Control Group
    for i, name in enumerate(df_filtered['Model Name']):
        if f"Improved" in name:
            bars[i].set_color('coral')
        
        if f"ablation" in name:
            bars[i].set_color('lightgreen')
    
    plt.title(f'{model_type} Models - Ablation Study Comparison')
    plt.xlabel('Model Name')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.ylim(min(df_filtered['Best Val Acc (%)'].min()-5,50), df_filtered['Best Val Acc (%)'].max()+2)
    plt.tight_layout()

    plot_path=os.path.join(output_dir, f"{model_type}_ablation_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Ablation comparison plot for {model_type} saved to {plot_path}")

def main():
    """Load results and generate report and plots."""
    all_results=load_results(RESULTS_FILE)
    if all_results is None:
        return
    
    # Generate summary table
    df_summary=generate_summary_table(all_results)

    # 2. Plot learning curves for each model
    print("\n--- Generating learning curves for each model ---")
    # Choose model for NN and CNN respectively
    for model_name in ['NN_Baseline', 'NN_Improved_NoReg', 'CNN_Baseline', 'CNN_ablation_AvgPool']:
        if model_name in all_results and 'History' in all_results[model_name]:
            plot_learning_curves(model_name, all_results[model_name]['History'])
    
    # Draw ablation comparison bar chart
    print("\n--- Generating ablation comparison plots ---")
    plot_ablation_comparison(df_summary,'NN')
    plot_ablation_comparison(df_summary,'CNN')

if __name__=='__main__':
    main()