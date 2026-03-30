import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def create_visualizations(scores_file, output_dir):
    """
    Generates plots for the BADASP scores.
    """
    print(f"Loading scores from {scores_file}...")
    if not os.path.exists(scores_file):
        print(f"Error: Scores file {scores_file} not found.")
        return
        
    df = pd.read_csv(scores_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Global distribution of BADASP scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['BADASP_Score'], bins=50, kde=True, color='coral')
    plt.title('Distribution of BADASP Scores')
    plt.xlabel('BADASP Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Maximum score per site
    max_scores = df.groupby('Site')['BADASP_Score'].max().reset_index()
    
    plt.figure(figsize=(14, 5))
    plt.stem(max_scores['Site'], max_scores['BADASP_Score'], basefmt=" ", markerfmt="o")
    plt.title('Maximum BADASP Score per Alignment Site')
    plt.xlabel('Alignment Site Position')
    plt.ylabel('Max BADASP Score')
    plt.grid(True, alpha=0.3)
    
    # Annotate top 5 sites
    top_sites = max_scores.nlargest(5, 'BADASP_Score')
    for _, row in top_sites.iterrows():
        plt.annotate(f"{int(row['Site'])}", 
                     xy=(row['Site'], row['BADASP_Score']), 
                     xytext=(0, 5), textcoords='offset points', ha='center',
                     fontsize=9, fontweight='bold', color='red')
                     
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'max_score_per_site.png'), dpi=300)
    plt.close()
    
    # Isolate functional switches (>95th percentile)
    if 'Is_Functional_Switch' in df.columns:
        switches = df[df['Is_Functional_Switch']]
    else:
        threshold = df['BADASP_Score'].quantile(0.95)
        switches = df[df['BADASP_Score'] > threshold]
        
    print(f"\nFound {len(switches)} switch events.")
    
    # Save the top switches
    top_events = df.nlargest(10, 'BADASP_Score')
    top_events.to_csv(os.path.join(output_dir, 'top_events.csv'), index=False)
    
    # Generate ChimeraX script for 3D mapping
    unique_sites = switches['Site'].unique()
    cxc_path = os.path.join(output_dir, 'highlight_switches.cxc')
    with open(cxc_path, 'w') as f:
        f.write("# ChimeraX script to highlight BADASP >95th percentile functional switches\n")
        f.write("# Usage: open your_structure.pdb, then run this script (open highlight_switches.cxc)\n\n")
        f.write("color all #e0e0e0\n") # Color everything light gray
        if len(unique_sites) > 0:
            site_str = ",".join(str(int(s)) for s in unique_sites)
            f.write(f"color :{site_str} red\n") # Color switches red
            f.write(f"show :{site_str} surface\n")
            
    print(f"Generated ChimeraX 3D mapping script at {cxc_path}")
    print(f"\nSaved plots and output data to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BADASP scores.")
    parser.add_argument("--scores", type=str, default="results/badasp_scores.csv", help="Input BADASP scores CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for plots")
    
    args = parser.parse_args()
    create_visualizations(args.scores, args.output_dir)
