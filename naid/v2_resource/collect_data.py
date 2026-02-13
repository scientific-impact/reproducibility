import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import AutoTokenizer

# Import your custom dataset class
# adjusting the import path based on where you place this script relative to dataset.py
try:
    from v2_resource.NAIDv2.dataset import PairwisePaperDataset
except ImportError:
    # Fallback if the script is in the same directory as dataset.py
    from dataset import PairwisePaperDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Pairwise Data Distribution")
    
    # File paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to your raw train.csv")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Where to save plots and csv")
    
    # Dataset Config (Must match your train.py config to be accurate)
    parser.add_argument("--gt_field", type=str, default="RTS") # Or score_gauss_mix
    parser.add_argument("--max_pairs", type=int, default=10000)
    parser.add_argument("--min_diff", type=float, default=0.05)
    
    # Grouping Strategy
    parser.add_argument("--group_by_cluster_year", type=str, default="False") # "True" or "False"
    parser.add_argument("--group_keys", type=str, default="pub_year,cluster_cat")
    
    # Sampling Strategy
    parser.add_argument("--bucket_edges", type=str, default='[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,"inf"]')
    parser.add_argument("--target_ratio", type=str, default="[0.03,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.27,0.00]")
    
    args = parser.parse_args()
    
    # Helper to parse bool
    if args.group_by_cluster_year.lower() in ['true', '1', 'yes']:
        args.group_by_cluster_year = True
    else:
        args.group_by_cluster_year = False
        
    return args

# --- Helpers copied/adapted from your train.py to parse args correctly ---
def _parse_list(s):
    if s is None: return None
    try:
        return json.loads(s)
    except:
        return [float(x) for x in s.split(',') if x.strip()]

def _parse_bucket_edges(s):
    lst = _parse_list(s)
    out = []
    for x in lst:
        if isinstance(x, str) and 'inf' in x.lower():
            out.append(float('inf'))
        else:
            out.append(float(x))
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Load Data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # We need a dummy tokenizer just to initialize the class
    # (We won't actually tokenize text, so any tokenizer works)
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 

    print("Building Pairwise Dataset...")
    # Parse list arguments
    bucket_edges = _parse_bucket_edges(args.bucket_edges)
    target_ratio = _parse_list(args.target_ratio)
    group_keys = tuple(args.group_keys.split(','))

    # Instantiate the dataset
    ds = PairwisePaperDataset(
        data=df,
        tokenizer=tokenizer,
        max_pairs=args.max_pairs,
        gt_field=args.gt_field,
        group_by_cluster_year=args.group_by_cluster_year,
        group_keys=group_keys,
        min_diff=args.min_diff,
        bucket_edges=bucket_edges,
        target_ratio=target_ratio,
        verbose=True 
    )

    # --- EXTRACT DATA ---
    # Instead of using __getitem__ (which tokenizes and is slow),
    # we directly access ds.pairs which contains (row_a, row_b, diff)
    
    print(f"Extracting {len(ds.pairs)} pairs for analysis...")
    
    records = []
    for i, (row_a, row_b, diff) in enumerate(ds.pairs):
        # Logic to match __getitem__ swapping behavior if needed, 
        # but for analysis we just want raw values
        
        score_a = float(row_a[args.gt_field])
        score_b = float(row_b[args.gt_field])
        
        # Identify which bucket A and B belong to (0-9)
        # Assuming scores are 0-1. If not, adjust math.
        bucket_a = int(score_a * 10) if score_a < 1.0 else 9
        bucket_b = int(score_b * 10) if score_b < 1.0 else 9
        
        records.append({
            'pair_id': i,
            'score_a': score_a,
            'score_b': score_b,
            'diff': diff,
            'abs_diff': abs(diff),
            'bucket_a': bucket_a,
            'bucket_b': bucket_b,
            'is_medium_pair': (3 <= bucket_a <= 6) and (3 <= bucket_b <= 6),
            'title_a': row_a.get('title', ''),
            'title_b': row_b.get('title', '')
        })
    
    analysis_df = pd.DataFrame(records)
    
    # --- VISUALIZATION ---
    
    # 1. Distribution of Differences (The "Delta")
    plt.figure(figsize=(10, 6))
    sns.histplot(analysis_df['abs_diff'], bins=30, kde=True)
    plt.title(f'Distribution of Score Differences (N={len(analysis_df)})')
    plt.xlabel('Absolute Difference |Score A - Score B|')
    plt.axvline(args.min_diff, color='r', linestyle='--', label=f'Min Diff ({args.min_diff})')
    plt.legend()
    plt.savefig(f"{args.output_dir}/01_diff_distribution.png")
    plt.close()

    # 2. Heatmap of Pair Interactions (Which buckets are being compared?)
    # We create a 10x10 matrix counting how often Bucket X is paired with Bucket Y
    heatmap_data = np.zeros((10, 10))
    for _, row in analysis_df.iterrows():
        i, j = int(row['bucket_a']), int(row['bucket_b'])
        heatmap_data[i][j] += 1
        heatmap_data[j][i] += 1 # Symmetric
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='viridis')
    plt.title('Pairing Heatmap: Which buckets are compared against each other?')
    plt.xlabel('Bucket')
    plt.ylabel('Bucket')
    plt.savefig(f"{args.output_dir}/02_pairing_heatmap.png")
    plt.close()

    # 3. Scatter Plot: Score A vs Score B
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=analysis_df, x='score_a', y='score_b', alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--') # Identity line
    plt.title('Score A vs Score B Correlation in Pairs')
    plt.savefig(f"{args.output_dir}/03_score_scatter.png")
    plt.close()

    # --- STATISTICS ---
    print("\n" + "="*30)
    print("ANALYSIS SUMMARY")
    print("="*30)
    print(f"Total Pairs: {len(analysis_df)}")
    print(f"Avg Diff: {analysis_df['abs_diff'].mean():.4f}")
    
    medium_pairs = analysis_df[analysis_df['is_medium_pair']]
    print(f"Medium vs Medium Pairs (Buckets 3-6): {len(medium_pairs)} ({len(medium_pairs)/len(analysis_df):.1%})")
    print(f"  -> Avg Diff in Medium Pairs: {medium_pairs['abs_diff'].mean():.4f}")
    
    # Save Data
    csv_path = f"{args.output_dir}/sampled_pairs_analysis.csv"
    analysis_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved raw pair data to: {csv_path}")
    print(f"✅ Saved plots to: {args.output_dir}/")

if __name__ == "__main__":
    main()