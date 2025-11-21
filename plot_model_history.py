import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def plot_history(output_path='model_comparison_plot.png'):
    # Find all result_*.xlsx files
    result_files = glob.glob('result_*.xlsx')
    
    if not result_files:
        print("Error: No result_*.xlsx files found.")
        return

    print(f"Found {len(result_files)} result files: {result_files}")

    # Combine all results into a single DataFrame
    all_data = []
    for file in result_files:
        try:
            # Extract model name from filename (e.g., result_LSTM.xlsx -> LSTM)
            model_name = file.replace('result_', '').replace('.xlsx', '')
            
            df = pd.read_excel(file)
            df['Model'] = model_name
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("Error: No valid data found in result files.")
        return
    
    # Combine all DataFrames
    df = pd.concat(all_data, ignore_index=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison History', fontsize=16)

    # Metrics to plot
    metrics = [
        ('Train Loss', axes[0, 0]),
        ('Train Acc', axes[0, 1]),
        ('Test Loss', axes[1, 0]),
        ('Test Acc', axes[1, 1])
    ]

    # Plot each metric
    for metric, ax in metrics:
        sns.lineplot(data=df, x='Epoch', y=metric, hue='Model', ax=ax, marker='o', markersize=4)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        # ax.legend(title='Model')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_history()
