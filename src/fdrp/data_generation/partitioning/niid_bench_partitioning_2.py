"""
Based on the approach from: https://github.com/Xtra-Computing/NIID-Bench
Mostly AI adapted
"""
from __future__ import annotations
from typing import Optional, Dict, List
import numpy as np
import pandas as pd


def partition_dirichlet_2d(
    df: pd.DataFrame,
    n_clients: int,
    beta_L: float,
    beta_Q: float,
    label_column: str,
    client_column: str = "Client",
    min_size: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Joint Dirichlet partitioning controlling both:

        beta_L : label skew
        beta_Q : quantity skew

    Algorithm
    ----------
    1. Sample client sizes via Dirichlet(beta_Q)
    2. Allocate labels via Dirichlet(beta_L) under capacity constraints

    Returns
    -------
    df with reassigned client_column
    """

    if seed is not None:
        np.random.seed(seed)

    N = len(df)

    if n_clients * min_size > N:
        raise ValueError(
            f"Dataset too small for min_size constraint "
            f"(N={N}, n_clients={n_clients}, min_size={min_size})"
        )

    df = df.copy()

    # ------------------------------------------------------------
    # STEP 1: Sample client sizes (quantity skew)
    # ------------------------------------------------------------

    proportions = np.random.dirichlet([beta_Q] * n_clients)

    remaining = N - n_clients * min_size

    extra_sizes = np.random.multinomial(remaining, proportions)

    client_sizes = extra_sizes + min_size

    capacity = {i: int(size) for i, size in enumerate(client_sizes)}

    # ------------------------------------------------------------
    # STEP 2: Allocate samples label-by-label
    # ------------------------------------------------------------

    client_assignments: Dict[int, List[int]] = {i: [] for i in range(n_clients)}

    labels = df[label_column].unique()

    for label in labels:

        indices = df.index[df[label_column] == label].to_numpy().copy()

        np.random.shuffle(indices)

        n_label = len(indices)

        proportions = np.random.dirichlet([beta_L] * n_clients)

        desired = np.random.multinomial(n_label, proportions)

        start = 0

        for client_id in range(n_clients):

            available = capacity[client_id]

            give = min(desired[client_id], available)

            if give > 0:

                assigned = indices[start:start + give]

                client_assignments[client_id].extend(assigned)

                capacity[client_id] -= give

                start += give

        # fallback allocation if capacity limits blocked some samples
        remaining_indices = indices[start:]

        for idx in remaining_indices:

            available_clients = [c for c in range(n_clients) if capacity[c] > 0]

            if not available_clients:
                break

            chosen = np.random.choice(available_clients)

            client_assignments[chosen].append(idx)

            capacity[chosen] -= 1

    # ------------------------------------------------------------
    # STEP 3: Write assignments to dataframe
    # ------------------------------------------------------------

    for client_id, indices in client_assignments.items():
        df.loc[indices, client_column] = client_id + 1

    return df

def partition_dirichlet_label_quantity(
    df,
    n_clients,
    beta_L,
    beta_Q,
    label_column,
    client_column="Client",
    min_size=10,
    seed=None,
):
    """
    Backwards-compatible wrapper for the new partition_dirichlet_2d function.

    This keeps the existing pipeline intact while the new implementation
    lives in partition_dirichlet_2d.
    """
    return partition_dirichlet_2d(
        df=df,
        n_clients=n_clients,
        beta_L=beta_L,
        beta_Q=beta_Q,
        label_column=label_column,
        client_column=client_column,
        min_size=min_size,
        seed=seed,
    )
import numpy as np
import pandas as pd


def print_partition_statistics(df, label_column, client_column="Client"):
    print("\nPartition Statistics")
    counts = df.groupby(client_column)[label_column].value_counts().unstack(fill_value=0)
    print(counts)


def compute_partition_heterogeneity_metrics(df, label_column, client_column="Client"):
    label_dist = (
        df.groupby(client_column)[label_column]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    global_dist = df[label_column].value_counts(normalize=True)

    kl_list = []

    for _, row in label_dist.iterrows():
        p = row.values + 1e-12
        q = global_dist.reindex(label_dist.columns, fill_value=0).values + 1e-12
        kl = np.sum(p * np.log(p / q))
        kl_list.append(kl)

    return {
        "mean_kl_divergence": float(np.mean(kl_list)),
        "std_kl_divergence": float(np.std(kl_list)),
    }


def print_quantity_skew_statistics(df, client_column="Client"):
    print("\nQuantity Skew Statistics")
    counts = df[client_column].value_counts().sort_index()
    print(counts)


def compute_quantity_skew_metrics(df, client_column="Client"):
    counts = df[client_column].value_counts().values

    return {
        "min_client_size": int(np.min(counts)),
        "max_client_size": int(np.max(counts)),
        "std_client_size": float(np.std(counts)),
    }