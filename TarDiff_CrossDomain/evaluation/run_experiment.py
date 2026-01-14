# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
All-in-one experiment script: specify data ratios, auto-complete data prep, training, evaluation.

Usage:
    python run_experiment.py --real_ratio 1.0 --synth_ratio 0.0 --name baseline
    python run_experiment.py --real_ratio 0.0 --synth_ratio 1.0 --name TSTR
    python run_experiment.py --real_ratio 1.0 --synth_ratio 1.0 --name TSRTR
    python run_experiment.py --real_ratio 0.5 --synth_ratio 0.5 --name mixed_50_50
"""

import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, rnn_type="gru", dropout=0.2, num_classes=2):
        super().__init__()
        RNN = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        output_dim = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        # return logits
        return self.fc(out[:, -1, :]).squeeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, stats=None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        if stats is None:
            self.mean = self.data.mean(dim=(0, 1), keepdim=True)
            self.std = self.data.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
        else:
            self.mean, self.std = stats
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = (self.data[idx] - self.mean.squeeze()) / self.std.squeeze()
        return x, self.labels[idx]


def load_pickle(path):
    """Load pickle file and convert to RNN format (N, T, C)"""
    with open(path, 'rb') as f:
        data, labels = pickle.load(f)
    data = np.array(data)
    labels = np.array(labels)
    if data.ndim == 3 and data.shape[1] < data.shape[2]:
        data = data.transpose(0, 2, 1)
    return data, labels


def prepare_data(real_path, synth_path, real_ratio, synth_ratio, seed=42):
    """Mix real and synthetic data by ratio"""
    np.random.seed(seed)
    real_data, real_labels = load_pickle(real_path)
    synth_data, synth_labels = load_pickle(synth_path)
    n_real = len(real_data)
    
    n_real_use = int(n_real * real_ratio)
    if n_real_use > 0:
        idx = np.random.choice(n_real, n_real_use, replace=real_ratio > 1)
        sampled_real = real_data[idx]
        sampled_real_labels = real_labels[idx]
    else:
        sampled_real = np.array([]).reshape(0, *real_data.shape[1:])
        sampled_real_labels = np.array([])
    
    n_synth_use = int(n_real * synth_ratio)
    if n_synth_use > 0:
        replace = n_synth_use > len(synth_data)
        idx = np.random.choice(len(synth_data), n_synth_use, replace=replace)
        sampled_synth = synth_data[idx]
        sampled_synth_labels = synth_labels[idx]
    else:
        sampled_synth = np.array([]).reshape(0, *real_data.shape[1:])
        sampled_synth_labels = np.array([])
    
    if len(sampled_real) > 0 and len(sampled_synth) > 0:
        data = np.concatenate([sampled_real, sampled_synth])
        labels = np.concatenate([sampled_real_labels, sampled_synth_labels])
    elif len(sampled_real) > 0:
        data, labels = sampled_real, sampled_real_labels
    else:
        data, labels = sampled_synth, sampled_synth_labels
    
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def train_and_evaluate(train_data, train_labels, val_path, test_path, args):
    """Train model and evaluate"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_data, val_labels = load_pickle(val_path)
    test_data, test_labels = load_pickle(test_path)
    
    # Detect number of classes
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    num_classes = len(np.unique(all_labels))
    print(f"Detected {num_classes} classes")

    train_ds = TimeSeriesDataset(train_data, train_labels)
    val_ds = TimeSeriesDataset(val_data, val_labels, (train_ds.mean, train_ds.std))
    test_ds = TimeSeriesDataset(test_data, test_labels, (train_ds.mean, train_ds.std))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    model = RNNClassifier(args.input_dim, args.hidden_dim, args.num_layers, args.rnn_type, num_classes=num_classes).to(device)
    
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_auroc = 0
    best_state = None
    patience = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            
            if num_classes == 2:
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y.long())
                
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_logits, val_labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                val_logits.append(logits.cpu())
                val_labels_list.append(y)
        
        logits_cat = torch.cat(val_logits)
        val_labels_np = torch.cat(val_labels_list).numpy()
        
        if num_classes == 2:
            val_probs = torch.sigmoid(logits_cat).numpy()
            val_auroc = roc_auc_score(val_labels_np, val_probs)
        else:
            val_probs = torch.softmax(logits_cat, dim=1).numpy()
            try:
                val_auroc = roc_auc_score(val_labels_np, val_probs, multi_class='ovr')
            except ValueError:
                val_auroc = 0.5  # Fallback if only one class present in batch
        
        print(f"Epoch {epoch:02d} | Val AUROC: {val_auroc:.4f}", end="")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = model.state_dict().copy()
            patience = 0
            print(" *")
        else:
            patience += 1
            print()
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    test_logits, test_labels_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_logits.append(model(x.to(device)).cpu())
            test_labels_list.append(y)
    
    logits_cat = torch.cat(test_logits)
    labels_np = torch.cat(test_labels_list).numpy()
    
    if num_classes == 2:
        probs = torch.sigmoid(logits_cat).numpy()
        preds = (probs > 0.5).astype(int)
        auroc = roc_auc_score(labels_np, probs)
        auprc = average_precision_score(labels_np, probs)
    else:
        probs = torch.softmax(logits_cat, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        try:
            auroc = roc_auc_score(labels_np, probs, multi_class='ovr')
        except ValueError:
            auroc = 0.5
        # AUPRC for multi-class is tricky, usually macro/micro average. 
        # Since average_precision_score handles multilabel but needs one-hot for multiclass,
        # we'll simplify or skip AUPRC for multiclass to avoid complexity, or use macro average manually.
        # For simplicity, let's use a weighted one-vs-rest approach if possible, or just set to 0.
        # sklearn's average_precision_score works if y_true is binarized.
        
        from sklearn.preprocessing import label_binarize
        y_onehot = label_binarize(labels_np, classes=np.arange(num_classes))
        auprc = average_precision_score(y_onehot, probs, average="macro")
    
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy_score(labels_np, preds),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="All-in-one experiment script")
    parser.add_argument("--real_data", "-r", type=str, default="../TarDiff_CrossDomain/data/processed/nasdaq/train_tuple.pkl")
    parser.add_argument("--synth_data", "-s", type=str, default="../TarDiff_CrossDomain/data/processed/nasdaq/synt_tardiff_noise_rnn_train_no_guidance_ms20k.pkl")
    parser.add_argument("--val_data", "-v", type=str, default="../TarDiff_CrossDomain/data/processed/nasdaq/val_tuple.pkl")
    parser.add_argument("--test_data", "-t", type=str, default="../TarDiff_CrossDomain/data/processed/nasdaq/test_tuple.pkl")
    parser.add_argument("--real_ratio", type=float, required=True, help="Real data ratio")
    parser.add_argument("--synth_ratio", type=float, required=True, help="Synthetic data ratio")
    parser.add_argument("--name", "-n", type=str, default=None, help="Experiment name")
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension (auto-detect)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--rnn_type", type=str, default="gru")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", "-o", type=str, default="./results")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.name is None:
        args.name = f"r{args.real_ratio}_s{args.synth_ratio}"
    
    print("=" * 60)
    print(f"Experiment: {args.name}")
    print(f"Config: real_ratio={args.real_ratio}, synth_ratio={args.synth_ratio}")
    print("=" * 60)
    
    print("\n[Step 1] Preparing data...")
    train_data, train_labels = prepare_data(args.real_data, args.synth_data, args.real_ratio, args.synth_ratio, args.seed)
    print(f"Training data: {train_data.shape}, label distribution: {dict(Counter(train_labels))}")
    
    if args.input_dim is None:
        args.input_dim = train_data.shape[-1]
        print(f"Auto-detected input dim: {args.input_dim}")
    
    print("\n[Step 2] Training model...")
    metrics = train_and_evaluate(train_data, train_labels, args.val_data, args.test_data, args)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "name": args.name,
        "real_ratio": args.real_ratio,
        "synth_ratio": args.synth_ratio,
        "train_size": len(train_data),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / f"{args.name}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved: {output_dir / args.name}.json")


if __name__ == "__main__":
    main()
