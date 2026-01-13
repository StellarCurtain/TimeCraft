"""
Create a stratified subset from training data for faster debugging.

Usage:
    python create_subset.py --input train_tuple.pkl --output train_tuple_50.pkl --ratio 0.5
    python create_subset.py --ratio 0.1   # 10% subset
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import Counter


def create_subset(input_path: str, output_path: str, ratio: float = 0.05, seed: int = 42):
    """Create a stratified subset preserving class ratios."""
    np.random.seed(seed)
    
    print(f"Loading data: {input_path}")
    with open(input_path, 'rb') as f:
        data, labels = pickle.load(f)
    
    print(f"Original shape: {data.shape}")
    print(f"Original labels: {dict(Counter(labels))}")
    
    n_total = len(data)
    unique_labels = np.unique(labels)
    subset_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_label_subset = int(len(label_indices) * ratio)
        n_label_subset = max(n_label_subset, min(100, len(label_indices)))
        selected = np.random.choice(label_indices, size=n_label_subset, replace=False)
        subset_indices.extend(selected)
    
    subset_indices = np.array(subset_indices)
    np.random.shuffle(subset_indices)
    
    subset_data = data[subset_indices]
    subset_labels = labels[subset_indices]
    
    print(f"\nSubset shape: {subset_data.shape}")
    print(f"Subset labels: {dict(Counter(subset_labels))}")
    print(f"Ratio: {len(subset_data) / n_total * 100:.2f}%")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump((subset_data, subset_labels), f)
    
    print(f"\nSaved to: {output_path}")
    return subset_data, subset_labels


def main():
    parser = argparse.ArgumentParser(description="Create stratified data subset")
    parser.add_argument("--input", "-i", type=str, default="../data/processed/nasdaq/train_tuple.pkl")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--ratio", "-r", type=float, default=0.05, help="Subset ratio (default: 0.05)")
    parser.add_argument("--seed", "-s", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        pct_str = f"{int(args.ratio * 100)}pct"
        args.output = input_path.parent / f"train_tuple_debug_{pct_str}.pkl"
    
    create_subset(args.input, args.output, args.ratio, args.seed)


if __name__ == "__main__":
    main()
