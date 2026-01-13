# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Preprocess NASDAQ stock data for TarDiff framework.

Input: Raw CSV stock data (Date, Open, High, Low, Close, Adj Close, Volume)
Output: pickle tuple (data, labels) where data.shape=(N, C, T), labels.shape=(N,)

Usage:
    python preprocess_nasdaq_for_tardiff.py --input_path data/raw/NASDAQ --output_path data/processed/nasdaq --seq_len 24 --pred_horizon 5
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NASDAQ data for TarDiff')
    parser.add_argument('--input_path', type=str, required=True, help='NASDAQ raw data path')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length (default: 24)')
    parser.add_argument('--pred_horizon', type=int, default=5, help='Prediction horizon in days (default: 5)')
    parser.add_argument('--min_date', type=str, default='2010-01-01', help='Minimum date filter')
    parser.add_argument('--min_length', type=int, default=1000, help='Minimum stock data length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio (default: 0.1)')
    parser.add_argument('--channels', type=str, default='Open,High,Low,Close,Volume', help='Channels to use')
    parser.add_argument('--normalize_per_stock', action='store_true', help='Normalize per stock')
    parser.add_argument('--use_returns', action='store_true', help='Use returns instead of prices')
    return parser.parse_args()


def load_stock_data(file_path: str, min_date: str = None) -> pd.DataFrame:
    """Load single stock data."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if min_date:
            df = df[df['Date'] >= min_date].reset_index(drop=True)
        return df
    except Exception:
        return None


def compute_label(df: pd.DataFrame, idx: int, seq_len: int, pred_horizon: int) -> int:
    """Compute label: 1 if price goes up in pred_horizon days, 0 otherwise."""
    end_idx = idx + seq_len - 1
    future_idx = end_idx + pred_horizon
    if future_idx >= len(df):
        return None
    current_close = df.iloc[end_idx]['Close']
    future_close = df.iloc[future_idx]['Close']
    return 1 if future_close > current_close else 0


def extract_sequences(df: pd.DataFrame, channels: List[str], seq_len: int, 
                      pred_horizon: int, stride: int = 1, use_returns: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sequences from stock data. Returns data (N, C, T) and labels (N,)."""
    sequences, labels = [], []
    
    for idx in range(0, len(df) - seq_len - pred_horizon + 1, stride):
        seq_data = df.iloc[idx:idx + seq_len][channels].values
        label = compute_label(df, idx, seq_len, pred_horizon)
        if label is None or np.isnan(seq_data).any():
            continue
        
        if use_returns:
            seq_data = np.diff(seq_data, axis=0) / (seq_data[:-1] + 1e-8)
            seq_data = np.vstack([np.zeros((1, seq_data.shape[1])), seq_data])
        
        sequences.append(seq_data.T)
        labels.append(label)
    
    if len(sequences) == 0:
        return None, None
    return np.array(sequences), np.array(labels)


def normalize_data(data: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """Normalize data. TarDiff will do its own normalization (centered_pit)."""
    stats = {}
    if method == 'zscore':
        mean = np.nanmean(data, axis=(0, 2), keepdims=True)
        std = np.nanstd(data, axis=(0, 2), keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        stats['mean'], stats['std'] = mean, std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=(0, 2), keepdims=True)
        max_val = np.nanmax(data, axis=(0, 2), keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        stats['min'], stats['max'] = min_val, max_val
    else:
        normalized = data
    return normalized, stats


def split_dataset(data: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8,
                  val_ratio: float = 0.1, shuffle: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split into train/val/test sets."""
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    return {
        'train': (data[indices[:train_end]], labels[indices[:train_end]]),
        'val': (data[indices[train_end:val_end]], labels[indices[train_end:val_end]]),
        'test': (data[indices[val_end:]], labels[indices[val_end:]])
    }


def save_dataset(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: str, stats: Dict = None):
    """Save dataset in TarDiff format: train_tuple.pkl, val_tuple.pkl, test_tuple.pkl."""
    os.makedirs(output_path, exist_ok=True)
    
    for split_name, (data, labels) in data_dict.items():
        file_path = os.path.join(output_path, f'{split_name}_tuple.pkl')
        data, labels = data.astype(np.float32), labels.astype(np.int64)
        with open(file_path, 'wb') as f:
            pickle.dump((data, labels), f)
        print(f"  Saved: {file_path}")
        print(f"    Shape: {data.shape}, Labels: {labels.shape}, Pos ratio: {labels.mean()*100:.2f}%")
    
    if stats:
        meta_path = os.path.join(output_path, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"  Saved meta: {meta_path}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NASDAQ Preprocessing for TarDiff")
    print("=" * 70)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Seq length: {args.seq_len}, Pred horizon: {args.pred_horizon}")
    print(f"Channels: {args.channels}")
    print()
    
    channels = [c.strip() for c in args.channels.split(',')]
    
    stocks_path = os.path.join(args.input_path, 'stocks')
    stock_files = list(Path(stocks_path).glob('*.csv'))
    print(f"Found {len(stock_files)} stock files")
    
    all_sequences, all_labels = [], []
    valid_stocks = 0
    
    for stock_file in tqdm(stock_files, desc="Processing"):
        df = load_stock_data(str(stock_file), args.min_date)
        if df is None or len(df) < args.min_length:
            continue
        if not all(col in df.columns for col in channels):
            continue
        
        sequences, labels = extract_sequences(df, channels, args.seq_len, args.pred_horizon,
                                              stride=args.seq_len, use_returns=args.use_returns)
        if sequences is not None and len(sequences) > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)
            valid_stocks += 1
    
    print(f"\nValid stocks: {valid_stocks}")
    
    if len(all_sequences) == 0:
        print("Error: No valid samples!")
        return
    
    data = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal samples: {len(data)}")
    print(f"Shape: {data.shape} (samples, channels, timesteps)")
    print(f"Positive ratio: {labels.mean()*100:.2f}%")
    
    stats = {
        'channels': channels, 'n_channels': len(channels), 'seq_len': args.seq_len,
        'pred_horizon': args.pred_horizon, 'n_samples': len(data), 'pos_ratio': float(labels.mean())
    }
    
    print("\nSplitting dataset...")
    data_splits = split_dataset(data, labels, args.train_ratio, args.val_ratio)
    for split_name, (split_data, split_labels) in data_splits.items():
        print(f"  {split_name}: {len(split_data)} samples (pos: {split_labels.mean()*100:.2f}%)")
    
    print("\nSaving...")
    save_dataset(data_splits, args.output_path, stats)
    
    guidance_path = os.path.join(args.output_path, 'guidance_tuple.pkl')
    with open(guidance_path, 'wb') as f:
        pickle.dump(data_splits['val'], f)
    print(f"  Saved guidance: {guidance_path}")
    
    print("\n" + "=" * 70)
    print(f"Done! Shape: ({len(channels)}, {args.seq_len})")
    print("=" * 70)


if __name__ == '__main__':
    main()
