"""
Compute per-class gradient norm statistics for a trained classifier.
Minority (hard) samples typically have larger gradient norms than majority (easy) samples.

Usage:
    # NASDAQ dataset
    python compute_grad_norm.py --data_path ../data/processed/nasdaq/train_tuple.pkl --model_path ../models/nasdaq_base/classifier_best.pt --input_dim 5 --num_classes 2 --max_samples 1000

    # Wafer dataset
    python compute_grad_norm.py --data_path ../data/processed/wafer/train_tuple.pkl --model_path ../models/wafer_base/classifier_best.pt --input_dim 1 --num_classes 2

"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "TarDiff"))
from classifier.model import RNNClassifier


def compute_per_class_grad_norm_stats(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
    max_samples: int = None,
) -> dict:
    """Compute gradient norm statistics grouped by class label."""
    # Must use train() mode for cuDNN RNN backward
    model.train()
    model.to(device)
    criterion.to(device)
    
    grad_norms = {}
    sample_count = 0
    
    for inputs, labels in tqdm(data_loader, desc="Computing gradient norms"):
        if max_samples and sample_count >= max_samples:
            break
            
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        inputs.requires_grad_(False)
        
        # RNNClassifier expects (B, T, F), data is (B, F, T)
        inputs_transposed = inputs.transpose(1, 2)
        outputs = model(inputs_transposed)
        
        # Handle binary vs multi-class
        if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.view(-1), labels.float()
            )
        else:
            loss = criterion(outputs, labels)
        
        grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=False, allow_unused=True
        )
        
        filtered_grads = [g for g in grads if g is not None]
        if not filtered_grads:
            continue
        
        # L2 norm: sqrt(sum(g^2))
        grad_norm_sq = sum(torch.sum(g ** 2) for g in filtered_grads)
        grad_norm = torch.sqrt(grad_norm_sq).item()
        
        label_val = labels.item() if labels.numel() == 1 else int(labels[0].item())
        grad_norms.setdefault(label_val, []).append(grad_norm)
        sample_count += 1
    
    # Compute statistics
    stats = {}
    print("\n" + "=" * 60)
    print("Gradient Norm Statistics")
    print("=" * 60)
    
    for label_val in sorted(grad_norms.keys()):
        norm_list = grad_norms[label_val]
        mean, std = np.mean(norm_list), np.std(norm_list)
        stats[label_val] = (mean, std, len(norm_list))
        print(f"Class {label_val}: {mean:.4f} ± {std:.4f}  (n={len(norm_list)})")
    
    if len(stats) >= 2:
        print("\n" + "-" * 60)
        print("Analysis")
        print("-" * 60)
        
        sorted_by_mean = sorted(stats.items(), key=lambda x: x[1][0])
        min_class, (min_mean, _, min_n) = sorted_by_mean[0]
        max_class, (max_mean, _, max_n) = sorted_by_mean[-1]
        ratio = max_mean / min_mean if min_mean > 0 else float('inf')
        
        print(f"Min grad norm: Class {min_class} (mean={min_mean:.4f}, n={min_n})")
        print(f"Max grad norm: Class {max_class} (mean={max_mean:.4f}, n={max_n})")
        print(f"Ratio: {ratio:.2f}x")
        
        if ratio > 2:
            print(f"\nClass {max_class} samples are harder to classify.")
            print(f"TarDiff's Influence Guidance will assign larger weights to them.")
    
    print("=" * 60)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute per-class gradient norm statistics")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Path to train_tuple.pkl")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to classifier_best.pt")
    parser.add_argument("--input_dim", type=int, required=True, help="Input feature dimension")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (default: 2)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden dimension (default: 256)")
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["lstm", "gru"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for quick test")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for stats (optional)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data: {args.data_path}")
    with open(args.data_path, "rb") as f:
        data, labels = pickle.load(f)
    
    print(f"Data shape: {data.shape}")
    print(f"Label distribution: {dict(Counter(labels))}")
    
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} ({count/total*100:.2f}%)")
    
    dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).long())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    print(f"\nLoading model: {args.model_path}")
    model_num_classes = 1 if args.num_classes == 2 else args.num_classes
    
    model = RNNClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        rnn_type=args.rnn_type,
        num_classes=model_num_classes,
        dropout=0.2
    )
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(args.device)
    
    print(f"Model: {args.rnn_type.upper()}, hidden_dim={args.hidden_dim}")
    
    criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
    
    stats = compute_per_class_grad_norm_stats(
        model=model, data_loader=data_loader, criterion=criterion,
        device=args.device, max_samples=args.max_samples
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(stats, f)
        print(f"\nStats saved to: {output_path}")
    
    return stats


if __name__ == "__main__":
    main()
