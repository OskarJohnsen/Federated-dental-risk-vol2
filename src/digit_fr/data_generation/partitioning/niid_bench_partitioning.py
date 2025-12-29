"""
Based on the approach from: https://github.com/Xtra-Computing/NIID-Bench
Mostly AI adapted
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

def partition_dirichlet_label(df: pd.DataFrame, n_clients: int, beta: float, label_column: str, client_column: str = "Client", seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    
    labels = df[label_column].values
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    label_to_class = {label: idx for idx, label in enumerate(unique_labels)}
    
    class_indices: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    df_index = df.index.tolist()
    for pos_idx, label in enumerate(labels):
        class_idx = label_to_class[label]
        class_indices[class_idx].append(df_index[pos_idx])
    
    client_assignments: Dict[int, List[int]] = {i: [] for i in range(1, n_clients + 1)}
    
    for class_idx in range(n_classes):
        indices = class_indices[class_idx]
        n_samples_class = len(indices)
        
        if n_samples_class == 0:
            continue
        
        alpha = np.full(n_clients, beta)
        proportions = np.random.dirichlet(alpha)
        
        n_samples_per_client = np.random.multinomial(n_samples_class, proportions)
        
        shuffled_indices = np.random.permutation(indices)
        
        start_idx = 0
        for client_id in range(1, n_clients + 1):
            n_assigned = n_samples_per_client[client_id - 1]
            end_idx = start_idx + n_assigned
            assigned_indices = shuffled_indices[start_idx:end_idx].tolist()
            client_assignments[client_id].extend(assigned_indices)
            start_idx = end_idx
    
    df_partitioned = df.copy()
    df_partitioned[client_column] = 0
    
    for client_id, indices in client_assignments.items():
        df_partitioned.loc[indices, client_column] = client_id
    
    return df_partitioned


def get_label_distribution_stats(df: pd.DataFrame, label_column: str, client_column: str = "Client") -> Dict[int, Dict[int, int]]:
    stats = {}
    for client_id in sorted(df[client_column].unique()):
        client_data = df[df[client_column] == client_id]
        label_counts = client_data[label_column].value_counts().to_dict()
        stats[int(client_id)] = label_counts
    return stats

def print_partition_statistics(df: pd.DataFrame, label_column: str, client_column: str = "Client",) -> None:
    stats = get_label_distribution_stats(df, label_column, client_column)
    
    all_labels = sorted(set(label for client_stats in stats.values() for label in client_stats.keys()))
    
    print(f"Partition Statistics (Label: {label_column}")
    print(f"{'Client':<10}", end="")
    for label in all_labels:
        print(f"{'Label=' + str(label):<15}", end="")
    
    for client_id in sorted(stats.keys()):
        client_stats = stats[client_id]
        total = sum(client_stats.values())
        print(f"{client_id:<10}", end="")
        for label in all_labels:
            count = client_stats.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            print(f"{count} ({pct:5.1f}%){'':<5}", end="")
        print(f"{total:<10}")
    
    global_counts = df[label_column].value_counts().sort_index()
    global_total = len(df)
    print(f"{'Global':<10}", end="")
    for label in all_labels:
        count = global_counts.get(label, 0)
        pct = (count / global_total * 100) if global_total > 0 else 0.0
        print(f"{count} ({pct:5.1f}%){'':<5}", end="")
    print(f"{global_total:<10}")
    print()

def compute_partition_heterogeneity_metrics(df: pd.DataFrame, label_column: str, client_column: str = "Client") -> Dict[str, float]:
    stats = get_label_distribution_stats(df, label_column, client_column)
    all_labels = sorted(set(label for client_stats in stats.values() for label in client_stats.keys()))
    
    client_entropies = []
    client_diversities = []
    label_counts_per_client = {label: [] for label in all_labels}
    
    for client_id in sorted(stats.keys()):
        client_stats = stats[client_id]
        total = sum(client_stats.values())
        
        if total == 0:
            continue

        probs = [client_stats.get(label, 0) / total for label in all_labels]
        probs = [p for p in probs if p > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        client_entropies.append(entropy)
        
        diversity = sum(1 for label in all_labels if client_stats.get(label, 0) > 0)
        client_diversities.append(diversity)
        
        for label in all_labels:
            label_counts_per_client[label].append(client_stats.get(label, 0))
    
    global_counts = df[label_column].value_counts().sort_index()
    global_total = len(df)
    global_probs = [global_counts.get(label, 0) / global_total for label in all_labels]
    global_probs = [p for p in global_probs if p > 0]
    global_entropy = -sum(p * np.log2(p) for p in global_probs)
    
    imbalance_ratios = []
    for label in all_labels:
        counts = label_counts_per_client[label]
        if len(counts) > 0 and max(counts) > 0:
            ratio = max(counts) / max(min(counts), 1)
            imbalance_ratios.append(ratio)
    
    avg_imbalance_ratio = np.mean(imbalance_ratios) if imbalance_ratios else 1.0
    
    return {
        "label_entropy_per_client": float(np.mean(client_entropies)) if client_entropies else 0.0,
        "label_entropy_global": float(global_entropy),
        "client_label_diversity": float(np.mean(client_diversities)) if client_diversities else 0.0,
        "label_imbalance_ratio": float(avg_imbalance_ratio),
    }

def partition_quantity(df: pd.DataFrame, n_clients: int, beta_qty: float, label_column: str, client_column: str = "Client", min_size: int = 1, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    
    total_samples = len(df)
    labels = df[label_column].values
    unique_labels = np.unique(labels)
    
    alpha = np.full(n_clients, beta_qty)
    proportions = np.random.dirichlet(alpha)
    
    sizes = np.round(proportions * total_samples).astype(int)
    
    size_diff = total_samples - sizes.sum()
    if size_diff != 0:
        if size_diff > 0:
            largest_indices = np.argsort(sizes)[-size_diff:]
            sizes[largest_indices] += 1
        else:
            largest_indices = np.argsort(sizes)[:abs(size_diff)]
            sizes[largest_indices] -= 1
    
    sizes = np.maximum(sizes, min_size)
    
    current_total = sizes.sum()
    if current_total != total_samples:
        size_diff = total_samples - current_total
        if size_diff > 0:
            largest_indices = np.argsort(sizes)[-size_diff:]
            sizes[largest_indices] += 1
        else:
            largest_indices = np.argsort(sizes)[::-1]
            for idx in largest_indices:
                if size_diff == 0:
                    break
                if sizes[idx] > min_size:
                    reduction = min(abs(size_diff), sizes[idx] - min_size)
                    sizes[idx] -= reduction
                    size_diff += reduction
    
    label_indices: Dict[int, List[int]] = {}
    for label in unique_labels:
        label_indices[label] = df[df[label_column] == label].index.tolist()
    
    client_assignments: Dict[int, List[int]] = {i: [] for i in range(1, n_clients + 1)}
    
    for label in unique_labels:
        indices = label_indices[label]
        n_samples_label = len(indices)
        
        if n_samples_label == 0:
            continue
        
        shuffled_indices = np.random.permutation(indices).tolist()
        
        label_proportions = sizes / sizes.sum()
        n_samples_per_client = np.round(label_proportions * n_samples_label).astype(int)
        
        label_diff = n_samples_label - n_samples_per_client.sum()
        if label_diff != 0:
            if label_diff > 0:
                largest_indices = np.argsort(n_samples_per_client)[-label_diff:]
                n_samples_per_client[largest_indices] += 1
            else:
                largest_indices = np.argsort(n_samples_per_client)[:abs(label_diff)]
                n_samples_per_client[largest_indices] -= 1
        
        start_idx = 0
        for client_id in range(1, n_clients + 1):
            n_assigned = n_samples_per_client[client_id - 1]
            end_idx = start_idx + n_assigned
            assigned_indices = shuffled_indices[start_idx:end_idx]
            client_assignments[client_id].extend(assigned_indices)
            start_idx = end_idx
    
    df_partitioned = df.copy()
    df_partitioned[client_column] = 0
    
    for client_id, indices in client_assignments.items():
        df_partitioned.loc[indices, client_column] = client_id
    
    return df_partitioned

def get_client_sizes(df: pd.DataFrame, client_column: str = "Client") -> Dict[int, int]:
    sizes = {}
    for client_id in sorted(df[client_column].unique()):
        sizes[int(client_id)] = len(df[df[client_column] == client_id])
    return sizes

def print_quantity_skew_statistics(df: pd.DataFrame, client_column: str = "Client") -> None:
    sizes = get_client_sizes(df, client_column)
    total_samples = len(df)
    
    print(f"Quantity Skew Statistics:")
    print(f"{'Client':<10} {'Size':<15} {'Percentage':<15} {'Cumulative %':<15}")
    print("-" * 55)
    
    sorted_clients = sorted(sizes.keys())
    cumulative = 0
    for client_id in sorted_clients:
        size = sizes[client_id]
        pct = (size / total_samples * 100) if total_samples > 0 else 0.0
        cumulative += pct
        print(f"{client_id:<10} {size:<15} {pct:>6.2f}%{'':<7} {cumulative:>6.2f}%{'':<7}")
    
    size_values = list(sizes.values())
    min_size = min(size_values)
    max_size = max(size_values)
    mean_size = np.mean(size_values)
    std_size = np.std(size_values)
    cv = (std_size / mean_size) if mean_size > 0 else 0.0
    imbalance_ratio = (max_size / min_size) if min_size > 0 else float('inf')
    
    print("-" * 55)
    print(f"{'Total':<10} {total_samples:<15}")
    print(f"\nSummary Statistics:")
    print(f"Min size: {min_size}")
    print(f"Max size: {max_size}")
    print(f"Mean size: {mean_size:.1f}")
    print(f"Std size: {std_size:.1f}")
    print(f"Coefficient of Variation (CV): {cv:.4f}")
    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
    print()

def compute_quantity_skew_metrics(df: pd.DataFrame, client_column: str = "Client") -> Dict[str, float]:
    sizes = get_client_sizes(df, client_column)
    size_values = np.array(list(sizes.values()))
    
    total_samples = len(df)
    mean_size = float(np.mean(size_values))
    std_size = float(np.std(size_values))
    min_size = int(np.min(size_values))
    max_size = int(np.max(size_values))
    
    cv = (std_size / mean_size) if mean_size > 0 else 0.0
    imbalance_ratio = (max_size / min_size) if min_size > 0 else float('inf')
    
    sorted_sizes = np.sort(size_values)
    n = len(sorted_sizes)
    cumsum = np.cumsum(sorted_sizes)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_sizes)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0.0
    
    return {
        "min_size": min_size,
        "max_size": max_size,
        "mean_size": mean_size,
        "std_size": std_size,
        "coefficient_of_variation": cv,
        "imbalance_ratio": imbalance_ratio,
        "gini_coefficient": float(gini),
    }