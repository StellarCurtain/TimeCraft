#!/usr/bin/env python
"""
Diffusion model generation quality evaluation.

Usage:
    python quality_evaluation.py -r real_data.pkl -s0 synth_alpha0.pkl -s1 synth_alpha1e5.pkl -o output_dir
"""

import argparse
import pickle as pkl
import json
import numpy as np
from pathlib import Path


def compute_mmd(X, Y, gamma=None):
    """Compute MMD (Maximum Mean Discrepancy) between two distributions."""
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) == 3:
        Y = Y.reshape(Y.shape[0], -1)
    
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    
    X = X[np.all(np.isfinite(X), axis=1)]
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    
    if len(X) == 0 or len(Y) == 0:
        return np.nan
    
    combined = np.concatenate([X, Y], axis=0)
    mean, std = combined.mean(axis=0, keepdims=True), combined.std(axis=0, keepdims=True) + 1e-8
    X, Y = (X - mean) / std, (Y - mean) / std
    
    max_samples = 2000
    if len(X) > max_samples:
        X = X[np.random.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[np.random.choice(len(Y), max_samples, replace=False)]
    
    n_x, n_y = len(X), len(Y)
    
    if gamma is None:
        dists = np.sum((X[:100, None] - X[None, :100]) ** 2, axis=-1)
        median_dist = np.median(dists)
        gamma = 1.0 / (median_dist if median_dist > 0 and np.isfinite(median_dist) else 1.0)
    
    def rbf_kernel(A, B, g):
        return np.exp(-g * np.maximum(np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T, 0))
    
    K_xx, K_yy, K_xy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    mmd_sq = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) + (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) - 2 * np.mean(K_xy)
    return np.sqrt(max(mmd_sq, 0))


def load_data(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    X, y = (data[0], data[1]) if isinstance(data, tuple) else (data, None)
    if len(X.shape) == 3 and X.shape[1] < X.shape[2]:
        X = X.transpose(0, 2, 1)
    return X.astype(np.float32), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real_data', type=str, required=True)
    parser.add_argument('-s0', '--synth_alpha0', type=str, required=True)
    parser.add_argument('-s1', '--synth_alpha1', type=str, required=True)
    parser.add_argument('-a', '--alpha_val', type=str, default='1e-5', help="Alpha value for display")
    parser.add_argument('-o', '--output_dir', type=str, default='./quality_results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_real, y_real = load_data(args.real_data)
    X_alpha0, y_alpha0 = load_data(args.synth_alpha0)
    X_alpha1, y_alpha1 = load_data(args.synth_alpha1)
    
    label_s1 = f"α={args.alpha_val}"
    
    print(f"Real: {X_real.shape}, α=0: {X_alpha0.shape}, {label_s1}: {X_alpha1.shape}")
    
    for name, y in [("Real", y_real), ("α=0", y_alpha0), (label_s1, y_alpha1)]:
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            print(f"{name}: " + ", ".join([f"Class {l}: {c} ({c/len(y)*100:.1f}%)" for l, c in zip(unique, counts)]))
    
    mmd_alpha0 = compute_mmd(X_real, X_alpha0)
    mmd_alpha1 = compute_mmd(X_real, X_alpha1)
    
    print(f"MMD(Real, α=0): {mmd_alpha0:.4f}, MMD(Real, {label_s1}): {mmd_alpha1:.4f}")
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({'mmd_alpha0': float(mmd_alpha0), f'mmd_{label_s1}': float(mmd_alpha1)}, f, indent=2)


if __name__ == '__main__':
    main()
