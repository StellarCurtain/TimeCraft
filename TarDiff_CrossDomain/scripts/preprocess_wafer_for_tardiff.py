# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Preprocess Wafer data for TarDiff framework.

Source: UCR Time Series Classification Archive
- Univariate time series, length 152
- Labels: 1=normal, -1=abnormal
- Class imbalance: ~10-12% abnormal

Output: pickle tuple (data, labels) where data.shape=(N, 1, T), labels.shape=(N,)

Usage:
    python preprocess_wafer_for_tardiff.py --input_path data/raw/Wafer --output_path data/processed/wafer --seq_len 24
"""

import os
import argparse
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Dict


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Wafer data for TarDiff')
    parser.add_argument('--input_path', type=str, required=True, help='Wafer raw data path')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--seq_len', type=int, default=152, help='Output sequence length (default: 152)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def load_wafer_txt(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCR format Wafer data (.txt). Returns data (N, T) and labels (N,)."""
    print(f"Loading: {file_path}")
    data_list, labels_list = [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 2:
                continue
            labels_list.append(float(values[0]))
            data_list.append([float(v) for v in values[1:]])
    
    data = np.array(data_list, dtype=np.float32)
    labels = (np.array(labels_list) == -1).astype(np.int64)  # -1=abnormal -> 1
    
    print(f"  Samples: {len(data)}, Length: {data.shape[1]}, Abnormal: {labels.sum()} ({labels.mean()*100:.2f}%)")
    return data, labels


def load_wafer_ts(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCR .ts format Wafer data. Returns data (N, T) and labels (N,)."""
    print(f"Loading: {file_path}")
    data_list, labels_list = [], []
    
    with open(file_path, 'r') as f:
        in_data_section = False
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('@'):
                if line == '@data':
                    in_data_section = True
                continue
            if not in_data_section or not line:
                continue
            if ':' in line:
                ts_str, label_str = line.rsplit(':', 1)
                data_list.append([float(v) for v in ts_str.split(',')])
                labels_list.append(float(label_str))
    
    data = np.array(data_list, dtype=np.float32)
    labels = (np.array(labels_list) == -1).astype(np.int64)
    
    print(f"  Samples: {len(data)}, Length: {data.shape[1]}, Abnormal: {labels.sum()} ({labels.mean()*100:.2f}%)")
    return data, labels


def split_train_val(data: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1, 
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train set into train/val with stratified sampling."""
    np.random.seed(seed)
    
    pos_idx, neg_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    n_val_pos, n_val_neg = int(len(pos_idx) * val_ratio), int(len(neg_idx) * val_ratio)
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    np.random.shuffle(val_idx)
    np.random.shuffle(train_idx)
    
    return data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]


def save_dataset(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: str, stats: Dict = None):
    """Save dataset in TarDiff format. Adds channel dimension: (N, T) -> (N, 1, T)."""
    os.makedirs(output_path, exist_ok=True)
    
    for split_name, (data, labels) in data_dict.items():
        file_path = os.path.join(output_path, f'{split_name}_tuple.pkl')
        data = data[:, np.newaxis, :].astype(np.float32)
        labels = labels.astype(np.int64)
        with open(file_path, 'wb') as f:
            pickle.dump((data, labels), f)
        print(f"  Saved: {file_path}")
        print(f"    Shape: {data.shape}, Labels: {labels.shape}, Abnormal: {labels.mean()*100:.2f}%")
    
    if stats:
        meta_path = os.path.join(output_path, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"  Saved meta: {meta_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("Wafer Preprocessing for TarDiff")
    print("=" * 70)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Target seq_len: {args.seq_len}")
    print()
    
    train_txt = os.path.join(args.input_path, 'Wafer_TRAIN.txt')
    test_txt = os.path.join(args.input_path, 'Wafer_TEST.txt')
    train_ts = os.path.join(args.input_path, 'Wafer_TRAIN.ts')
    test_ts = os.path.join(args.input_path, 'Wafer_TEST.ts')
    
    if os.path.exists(train_txt) and os.path.exists(test_txt):
        print("Using .txt format")
        train_data, train_labels = load_wafer_txt(train_txt)
        test_data, test_labels = load_wafer_txt(test_txt)
    elif os.path.exists(train_ts) and os.path.exists(test_ts):
        print("Using .ts format")
        train_data, train_labels = load_wafer_ts(train_ts)
        test_data, test_labels = load_wafer_ts(test_ts)
    else:
        raise FileNotFoundError(f"Wafer data not found at: {args.input_path}")
    
    original_seq_len = train_data.shape[1]
    
    if args.seq_len != original_seq_len:
        print(f"\nResampling: {original_seq_len} -> {args.seq_len}")
        from scipy.interpolate import interp1d
        
        def resample_data(data, target_len):
            n_samples, orig_len = data.shape
            x_orig, x_new = np.linspace(0, 1, orig_len), np.linspace(0, 1, target_len)
            resampled = np.zeros((n_samples, target_len), dtype=np.float32)
            for i in range(n_samples):
                resampled[i] = interp1d(x_orig, data[i], kind='linear')(x_new)
            return resampled
        
        train_data = resample_data(train_data, args.seq_len)
        test_data = resample_data(test_data, args.seq_len)
    
    print(f"\nSplitting train/val (ratio: {args.val_ratio})...")
    train_data, train_labels, val_data, val_labels = split_train_val(
        train_data, train_labels, args.val_ratio, args.seed)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} (abnormal: {train_labels.mean()*100:.2f}%)")
    print(f"  Val: {len(val_data)} (abnormal: {val_labels.mean()*100:.2f}%)")
    print(f"  Test: {len(test_data)} (abnormal: {test_labels.mean()*100:.2f}%)")
    
    stats = {
        'n_channels': 1, 'seq_len': args.seq_len, 'original_seq_len': original_seq_len,
        'n_train': len(train_data), 'n_val': len(val_data), 'n_test': len(test_data),
        'dataset': 'Wafer', 'source': 'UCR'
    }
    
    data_dict = {'train': (train_data, train_labels), 'val': (val_data, val_labels), 'test': (test_data, test_labels)}
    
    print("\nSaving...")
    save_dataset(data_dict, args.output_path, stats)
    
    guidance_path = os.path.join(args.output_path, 'guidance_tuple.pkl')
    with open(guidance_path, 'wb') as f:
        pickle.dump((val_data[:, np.newaxis, :].astype(np.float32), val_labels), f)
    print(f"  Saved guidance: {guidance_path}")
    
    print("\n" + "=" * 70)
    print(f"Done! Shape: (1, {args.seq_len})")
    print("=" * 70)


if __name__ == '__main__':
    main()
