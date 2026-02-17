import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def setup_style():
    """Sets up publication-quality plotting style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'lines.linewidth': 2.0,
        'figure.figsize': (8, 5),
        'savefig.dpi': 300
    })

def plot_training_curves(log_files, output_dir):
    """
    Plots training curves (Reward, Fidelity) from multiple log files.
    """
    data = []
    
    for log_file in log_files:
        path_parts = log_file.split(os.sep)
        label = "Run"
        if len(path_parts) > 1:
            dir_name = path_parts[-2] 
            if "noiseTrue" in dir_name:
                label = "Noise-Optimized"
            elif "noiseFalse" in dir_name:
                label = "Noise-Free"
            else:
                label = dir_name
        
        try:
            df = pd.read_json(log_file, lines=True)
            df['Condition'] = label
            # Moving average
            df['Avg_Fidelity_Smooth'] = df['avg_fidelity'].rolling(window=10).mean()
            df['Avg_Reward_Smooth'] = df['avg_reward'].rolling(window=10).mean()
            # New components
            if 'avg_leakage' in df.columns:
                df['Avg_Leak_Smooth'] = df['avg_leakage'].rolling(window=10).mean()
            if 'avg_bound' in df.columns:
                df['Avg_Bound_Smooth'] = df['avg_bound'].rolling(window=10).mean()
            if 'avg_time_cost' in df.columns:
                df['Avg_TimeC_Smooth'] = df['avg_time_cost'].rolling(window=10).mean()
                
            data.append(df)
        except Exception as e:
            print(f"Skipping {log_file}: {e}")
            
    if not data:
        print("No valid log files found.")
        return

    full_df = pd.concat(data, ignore_index=True)
    conditions = full_df['Condition'].unique()
    
    # Plot 1: Fidelity vs Iteration
    plt.figure()
    for cond in conditions:
        subset = full_df[full_df['Condition'] == cond]
        plt.plot(subset['iter'], subset['Avg_Fidelity_Smooth'], label=cond)
        # plt.plot(subset['iter'], subset['avg_fidelity'], alpha=0.3) # Noisy raw
        
    plt.title('Average Gate Fidelity over Training')
    plt.xlabel('Iteration (Batch)')
    plt.ylabel('Fidelity')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fidelity_curve.png"))
    plt.close()

    # Plot 2: Infidelity (Log Scale)
    plt.figure()
    for cond in conditions:
        subset = full_df[full_df['Condition'] == cond]
        infidelity = 1.0 - subset['Avg_Fidelity_Smooth']
        plt.plot(subset['iter'], infidelity, label=cond)
        
    plt.yscale('log')
    plt.title('Average Gate Infidelity (Log Scale)')
    plt.xlabel('Iteration')
    plt.ylabel('Infidelity (1 - F)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "infidelity_log_curve.png"))
    plt.close()
    
    # Plot 3: Reward
    plt.figure()
    for cond in conditions:
        subset = full_df[full_df['Condition'] == cond]
        plt.plot(subset['iter'], subset['Avg_Reward_Smooth'], label=cond)
        
    plt.title('Average Reward over Training')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    # Plot 4: Cost Decomposition
    # Plot Leakage, Boundary, Time for the last condition (or all? Too messy)
    # Let's plot Leakage separate
    plt.figure()
    for cond in conditions:
        subset = full_df[full_df['Condition'] == cond]
        if 'Avg_Leak_Smooth' in subset.columns:
            plt.plot(subset['iter'], subset['Avg_Leak_Smooth'], label=cond)
            
    plt.title('Average Leakage Cost (Eq 3 proxy)')
    plt.xlabel('Iteration')
    plt.ylabel('Leakage Cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "leakage_curve.png"))
    plt.close()

    # Plot 5: Boundary Cost
    plt.figure()
    for cond in conditions:
        subset = full_df[full_df['Condition'] == cond]
        if 'Avg_Bound_Smooth' in subset.columns:
            plt.plot(subset['iter'], subset['Avg_Bound_Smooth'], label=cond)
            
    plt.title('Average Boundary Cost (Power)')
    plt.xlabel('Iteration')
    plt.ylabel('Boundary Cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boundary_curve.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", help="List of log files (jsonl) or directory patterns")
    parser.add_argument("--out", type=str, default="plots", help="Output directory")
    args = parser.parse_args()
    
    # Resolve wildcards
    files = []
    if args.logs:
        for p in args.logs:
            files.extend(glob.glob(p, recursive=True))
    else:
        # Default search
        files = glob.glob("logs/**/training_log.jsonl", recursive=True)
        
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    plot_training_curves(files, args.out)
