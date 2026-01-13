# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Preprocess NASDAQ stock data for TarDiff framework.

Input: Raw CSV stock data (Date, Open, High, Low, Close, Adj Close, Volume)
Output: pickle tuple (data, labels) where data.shape=(N, C, T), labels.shape=(N,)

Usage:
    # Standard binary classification (up/down)
    python preprocess_nasdaq_for_tardiff.py --input_path data/raw/NASDAQ --output_path data/processed/nasdaq
    
    # Three-class classification (significant gain=1, neutral=0, significant loss=-1)
    python preprocess_nasdaq_for_tardiff.py --input_path data/raw/NASDAQ --output_path data/processed/nasdaq_extreme --label_mode extreme --extreme_percentile 5
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
    parser.add_argument('--use_returns', action='store_true', help='Use returns instead of prices')
    # Label mode
    parser.add_argument('--label_mode', type=str, default='binary', 
                        choices=['binary', 'extreme'],
                        help='Label mode: binary (up/down), extreme (3-class: gain/neutral/loss)')
    parser.add_argument('--extreme_percentile', type=float, default=5.0,
                        help='Percentile for extreme events (default: 5 means top/bottom 5%%)')
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
    """Compute binary label: 1 if price goes up, 0 otherwise."""
    end_idx = idx + seq_len - 1
    future_idx = end_idx + pred_horizon
    if future_idx >= len(df):
        return None
    current_close = df.iloc[end_idx]['Close']
    future_close = df.iloc[future_idx]['Close']
    return 1 if future_close > current_close else 0


def compute_return(df: pd.DataFrame, idx: int, seq_len: int, pred_horizon: int) -> float:
    """Compute return rate for the prediction horizon."""
    end_idx = idx + seq_len - 1
    future_idx = end_idx + pred_horizon
    if future_idx >= len(df):
        return None
    current_close = df.iloc[end_idx]['Close']
    future_close = df.iloc[future_idx]['Close']
    if current_close == 0:
        return None
    return (future_close - current_close) / current_close


def compute_extreme_label(return_rate: float, gain_threshold: float, loss_threshold: float) -> int:
    """
    Compute three-class label based on return rate.
    Returns:
        1: significant gain (return >= gain_threshold)
        0: neutral (in between)
        -1: significant loss (return <= loss_threshold)
    """
    if return_rate is None:
        return None
    if return_rate >= gain_threshold:
        return 1
    elif return_rate <= loss_threshold:
        return -1
    else:
        return 0


def extract_sequences(df: pd.DataFrame, channels: List[str], seq_len: int, 
                      pred_horizon: int, stride: int = 1, use_returns: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sequences with binary labels."""
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


def extract_sequences_with_returns(df: pd.DataFrame, channels: List[str], seq_len: int, 
                                   pred_horizon: int, stride: int = 1, 
                                   use_returns: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sequences with return rates for extreme labeling."""
    sequences, return_rates = [], []
    
    for idx in range(0, len(df) - seq_len - pred_horizon + 1, stride):
        seq_data = df.iloc[idx:idx + seq_len][channels].values
        ret = compute_return(df, idx, seq_len, pred_horizon)
        if ret is None or np.isnan(seq_data).any():
            continue
        
        if use_returns:
            seq_data = np.diff(seq_data, axis=0) / (seq_data[:-1] + 1e-8)
            seq_data = np.vstack([np.zeros((1, seq_data.shape[1])), seq_data])
        
        sequences.append(seq_data.T)
        return_rates.append(ret)
    
    if len(sequences) == 0:
        return None, None
    return np.array(sequences), np.array(return_rates)


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
    """Save dataset in TarDiff format."""
    os.makedirs(output_path, exist_ok=True)
    
    for split_name, (data, labels) in data_dict.items():
        file_path = os.path.join(output_path, f'{split_name}_tuple.pkl')
        data, labels = data.astype(np.float32), labels.astype(np.int64)
        with open(file_path, 'wb') as f:
            pickle.dump((data, labels), f)
        print(f"  Saved: {file_path}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = {int(l): int(c) for l, c in zip(unique_labels, counts)}
        print(f"    Shape: {data.shape}, Distribution: {label_dist}")
    
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
    print(f"Label mode: {args.label_mode}")
    if args.label_mode == 'extreme':
        print(f"Extreme percentile: {args.extreme_percentile}%")
    print()
    
    channels = [c.strip() for c in args.channels.split(',')]
    
    stocks_path = os.path.join(args.input_path, 'stocks')
    stock_files = list(Path(stocks_path).glob('*.csv'))
    print(f"Found {len(stock_files)} stock files")
    
    all_sequences, all_returns = [], []
    valid_stocks = 0
    
    for stock_file in tqdm(stock_files, desc="Processing"):
        df = load_stock_data(str(stock_file), args.min_date)
        if df is None or len(df) < args.min_length:
            continue
        if not all(col in df.columns for col in channels):
            continue
        
        if args.label_mode == 'binary':
            sequences, labels = extract_sequences(df, channels, args.seq_len, args.pred_horizon,
                                                  stride=args.seq_len, use_returns=args.use_returns)
            if sequences is not None and len(sequences) > 0:
                all_sequences.append(sequences)
                all_returns.append(labels)
                valid_stocks += 1
        else:
            sequences, returns = extract_sequences_with_returns(df, channels, args.seq_len, args.pred_horizon,
                                                                stride=args.seq_len, use_returns=args.use_returns)
            if sequences is not None and len(sequences) > 0:
                all_sequences.append(sequences)
                all_returns.append(returns)
                valid_stocks += 1
    
    print(f"\nValid stocks: {valid_stocks}")
    
    if len(all_sequences) == 0:
        print("Error: No valid samples!")
        return
    
    data = np.concatenate(all_sequences, axis=0)
    returns_or_labels = np.concatenate(all_returns, axis=0)
    
    print(f"\nTotal samples: {len(data)}")
    print(f"Shape: {data.shape}")
    
    if args.label_mode == 'binary':
        labels = returns_or_labels
        print(f"Positive ratio: {labels.mean()*100:.2f}%")
        stats = {
            'channels': channels, 'n_channels': len(channels), 'seq_len': args.seq_len,
            'pred_horizon': args.pred_horizon, 'n_samples': len(data), 
            'label_mode': 'binary', 'n_classes': 2
        }
    else:
        # Extreme mode: compute thresholds
        returns = returns_or_labels
        gain_threshold = np.percentile(returns, 100 - args.extreme_percentile)
        loss_threshold = np.percentile(returns, args.extreme_percentile)
        
        print(f"\nReturn statistics: mean={returns.mean()*100:.2f}%, std={returns.std()*100:.2f}%")
        print(f"Thresholds: gain >= {gain_threshold*100:.2f}%, loss <= {loss_threshold*100:.2f}%")
        
        # Compute labels
        labels = np.array([compute_extreme_label(r, gain_threshold, loss_threshold) for r in returns])
        
        # Remap: -1 -> 0, 0 -> 1, 1 -> 2 (for TarDiff compatibility)
        labels = labels + 1  # Now: loss=0, neutral=1, gain=2
        
        n_loss = (labels == 0).sum()
        n_neutral = (labels == 1).sum()
        n_gain = (labels == 2).sum()
        print(f"\nLabel distribution:")
        print(f"  Loss (0): {n_loss} ({n_loss/len(labels)*100:.2f}%)")
        print(f"  Neutral (1): {n_neutral} ({n_neutral/len(labels)*100:.2f}%)")
        print(f"  Gain (2): {n_gain} ({n_gain/len(labels)*100:.2f}%)")
        
        stats = {
            'channels': channels, 'n_channels': len(channels), 'seq_len': args.seq_len,
            'pred_horizon': args.pred_horizon, 'n_samples': len(data),
            'label_mode': 'extreme', 'n_classes': 3,
            'gain_threshold': float(gain_threshold),
            'loss_threshold': float(loss_threshold),
            'extreme_percentile': args.extreme_percentile
        }
    
    print("\nSplitting dataset...")
    data_splits = split_dataset(data, labels, args.train_ratio, args.val_ratio)
    
    print("\nSaving...")
    save_dataset(data_splits, args.output_path, stats)
    
    guidance_path = os.path.join(args.output_path, 'guidance_tuple.pkl')
    with open(guidance_path, 'wb') as f:
        pickle.dump(data_splits['val'], f)
    print(f"  Saved guidance: {guidance_path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
