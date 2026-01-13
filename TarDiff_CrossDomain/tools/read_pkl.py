#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
读取和检查 pickle 文件的工具

Usage: python tools/read_pkl.py <pkl_file_path> --verbose -n 5
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Tuple


def load_pkl(file_path: str) -> Any:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def analyze_tuple_data(data: Tuple[np.ndarray, np.ndarray], verbose: bool = False) -> dict:
    if not isinstance(data, tuple) or len(data) != 2:
        raise ValueError(f"期望 (data, labels) 元组，但得到: {type(data)}")
    
    data_array, labels = data[0], data[1]
    
    info = {
        'type': 'tuple',
        'data_shape': data_array.shape,
        'data_dtype': str(data_array.dtype),
        'labels_shape': labels.shape,
        'labels_dtype': str(labels.dtype),
        'n_samples': len(data_array),
    }
    
    if labels is not None and len(labels) > 0:
        unique_labels = np.unique(labels)
        info['n_classes'] = len(unique_labels)
        info['unique_labels'] = unique_labels.tolist()
        info['label_range'] = [int(labels.min()), int(labels.max())]
        
        label_dist, label_percent = {}, {}
        for label in unique_labels:
            count = np.sum(labels == label)
            label_dist[int(label)] = int(count)
            label_percent[int(label)] = float(count) / len(labels) * 100
        
        info['label_distribution'] = label_dist
        info['label_percent'] = label_percent
        info['has_nan_labels'] = bool(np.any(np.isnan(labels)))
        info['has_nan_data'] = bool(np.any(np.isnan(data_array)))
        if info['has_nan_labels']:
            info['nan_count'] = int(np.sum(np.isnan(labels)))
        if info['has_nan_data']:
            info['nan_data_count'] = int(np.sum(np.isnan(data_array)))
        
        info['data_min'] = float(np.nanmin(data_array))
        info['data_max'] = float(np.nanmax(data_array))
        info['data_mean'] = float(np.nanmean(data_array))
        info['data_std'] = float(np.nanstd(data_array))
    
    return info


def analyze_dict_data(data: dict, verbose: bool = False) -> dict:
    info = {
        'type': 'dict',
        'keys': list(data.keys()),
        'n_keys': len(data.keys()),
    }
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            info[f'{key}_shape'] = value.shape
            info[f'{key}_dtype'] = str(value.dtype)
            if verbose:
                info[f'{key}_min'] = float(np.nanmin(value))
                info[f'{key}_max'] = float(np.nanmax(value))
                info[f'{key}_mean'] = float(np.nanmean(value))
    return info


def analyze_data(data: Any, verbose: bool = False) -> dict:
    if isinstance(data, tuple):
        return analyze_tuple_data(data, verbose)
    elif isinstance(data, dict):
        return analyze_dict_data(data, verbose)
    elif isinstance(data, np.ndarray):
        return {
            'type': 'array',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'mean': float(np.nanmean(data)),
            'std': float(np.nanstd(data)),
        }
    else:
        return {'type': type(data).__name__, 'str_repr': str(data)[:200]}


def print_analysis(info: dict, file_path: str, verbose: bool = False, num_samples: int = 0, data: Any = None):
    print("=" * 80)
    print(f"Pickle 文件分析: {file_path}")
    print("=" * 80)
    
    if info['type'] == 'tuple':
        print(f"样本数量: {info['n_samples']:,}, 数据形状: {info['data_shape']}, 类型: {info['data_dtype']}")
        
        if 'n_classes' in info:
            print(f"类别数: {info['n_classes']}, 唯一标签值: {info['unique_labels']}")
            print("标签分布:")
            for label in sorted(info['label_distribution'].keys()):
                print(f"  类别 {label}: {info['label_distribution'][label]:,} ({info['label_percent'][label]:.2f}%)")
            
            if info.get('has_nan_labels'):
                print(f"⚠️ 标签中包含 NaN 值 ({info['nan_count']} 个)")
            if info.get('has_nan_data'):
                print(f"⚠️ 数据中包含 NaN 值 ({info['nan_data_count']} 个)")
            
            if verbose:
                print(f"数据统计: min={info['data_min']:.4f}, max={info['data_max']:.4f}, mean={info['data_mean']:.4f}, std={info['data_std']:.4f}")
        
        if info['n_classes'] == 1:
            print("⚠️ 警告: 数据中只有1个类别！建议检查数据预处理过程。")
    
    elif info['type'] == 'dict':
        print(f"键列表: {info['keys']}")
        for key in info['keys']:
            if f'{key}_shape' in info:
                line = f"  {key}: 形状={info[f'{key}_shape']}, 类型={info[f'{key}_dtype']}"
                if verbose and f'{key}_mean' in info:
                    line += f", mean={info[f'{key}_mean']:.4f}"
                print(line)
    
    elif info['type'] == 'array':
        print(f"形状: {info['shape']}, 类型: {info['dtype']}")
        print(f"min={info['min']:.4f}, max={info['max']:.4f}, mean={info['mean']:.4f}, std={info['std']:.4f}")
    
    else:
        print(f"数据类型: {info['type']}, 预览: {info.get('str_repr', 'N/A')}")
    
    if num_samples > 0 and data is not None and isinstance(data, tuple) and len(data) == 2:
        print(f"\n前 {num_samples} 条数据样本:")
        print("-" * 80)
        data_array, labels = data[0], data[1]
        for i in range(min(num_samples, len(data_array))):
            sample = data_array[i]
            print(f"样本 {i+1}: 标签={labels[i]}, 形状={sample.shape}")
            if len(sample.shape) == 2:
                for c in range(min(sample.shape[0], 5)):
                    ch = sample[c]
                    preview = f"[{', '.join(f'{v:.4f}' for v in ch[:4])}, ..., {', '.join(f'{v:.4f}' for v in ch[-4:])}]" if len(ch) > 8 else f"[{', '.join(f'{v:.4f}' for v in ch)}]"
                    print(f"  通道{c}: {preview}")
        print("-" * 80)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='读取和检查 pickle 文件')
    parser.add_argument('pkl_file', type=str, help='pickle 文件路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细统计')
    parser.add_argument('-n', '--num_samples', type=int, default=0, help='显示前N条样本')
    args = parser.parse_args()
    
    pkl_path = Path(args.pkl_file)
    if not pkl_path.is_absolute():
        current_dir = Path.cwd()
        if current_dir.name == 'TarDiff':
            alt_path = current_dir.parent / 'TarDiff_CrossDomain' / pkl_path
            pkl_path = alt_path if alt_path.exists() else current_dir / pkl_path
        else:
            pkl_path = current_dir / pkl_path
    pkl_path = pkl_path.resolve()
    
    try:
        print(f"正在加载: {pkl_path}")
        data = load_pkl(str(pkl_path))
        info = analyze_data(data, args.verbose)
        print_analysis(info, str(pkl_path), args.verbose, args.num_samples, data)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

