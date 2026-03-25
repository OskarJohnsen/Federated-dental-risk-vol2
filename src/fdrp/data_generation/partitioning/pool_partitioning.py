from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


def _sample_client_sizes(
    n_total: int,
    n_clients: int,
    beta_Q: float,
    min_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample exact client sizes using a Dirichlet + multinomial construction,
    while enforcing a minimum size for each client.
    """
    if n_clients * min_size > n_total:
        raise ValueError(
            f"min_size too large: n_clients * min_size = {n_clients * min_size}, "
            f"but dataset size = {n_total}"
        )

    size_proportions = rng.dirichlet([beta_Q] * n_clients)
    remaining = n_total - n_clients * min_size
    extra_sizes = rng.multinomial(remaining, size_proportions)
    client_sizes = extra_sizes + min_size
    return client_sizes.astype(int)


def _build_seed_matrix(
    client_sizes: np.ndarray,
    client_label_dists: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build a strictly positive seed matrix with desired row structure.
    Entry (k,c) is proportional to how much client k wants label c.
    """
    seed = client_sizes[:, None] * client_label_dists
    seed = np.maximum(seed, eps)
    return seed


def _ipf(
    seed_matrix: np.ndarray,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Iterative proportional fitting.
    Finds a positive matrix close to seed_matrix such that:
      - row sums equal row_sums
      - column sums equal col_sums
    """
    X = seed_matrix.astype(float).copy()

    row_sums = row_sums.astype(float)
    col_sums = col_sums.astype(float)

    if not np.isclose(row_sums.sum(), col_sums.sum()):
        raise ValueError(
            f"Row sums ({row_sums.sum()}) and column sums ({col_sums.sum()}) must match."
        )

    for _ in range(max_iter):
        # Scale rows
        current_row_sums = X.sum(axis=1)
        row_factors = np.divide(
            row_sums,
            current_row_sums,
            out=np.ones_like(row_sums),
            where=current_row_sums > 0,
        )
        X *= row_factors[:, None]

        # Scale columns
        current_col_sums = X.sum(axis=0)
        col_factors = np.divide(
            col_sums,
            current_col_sums,
            out=np.ones_like(col_sums),
            where=current_col_sums > 0,
        )
        X *= col_factors[None, :]

        # Check convergence
        row_err = np.max(np.abs(X.sum(axis=1) - row_sums))
        col_err = np.max(np.abs(X.sum(axis=0) - col_sums))
        if max(row_err, col_err) < tol:
            break
    else:
        raise RuntimeError("IPF did not converge.")

    return X


def _round_matrix_preserving_margins(
    X: np.ndarray,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
) -> np.ndarray:
    """
    Round a nonnegative real matrix X to integers while preserving
    exact row sums and column sums.

    Strategy:
      1) take floor
      2) distribute remaining units to cells with largest fractional parts,
         while respecting remaining row and column deficits
    """
    X_floor = np.floor(X).astype(int)
    frac = X - X_floor

    row_sums = row_sums.astype(int)
    col_sums = col_sums.astype(int)

    row_deficit = row_sums - X_floor.sum(axis=1)
    col_deficit = col_sums - X_floor.sum(axis=0)

    if row_deficit.sum() != col_deficit.sum():
        raise RuntimeError(
            f"Rounding failed: row deficit sum {row_deficit.sum()} "
            f"!= col deficit sum {col_deficit.sum()}."
        )

    X_int = X_floor.copy()

    # Sort all cells by descending fractional part
    cells: List[Tuple[int, int, float]] = [
        (i, j, frac[i, j])
        for i in range(X.shape[0])
        for j in range(X.shape[1])
    ]
    cells.sort(key=lambda t: t[2], reverse=True)

    total_to_add = int(row_deficit.sum())

    while total_to_add > 0:
        progress = False
        for i, j, _ in cells:
            if row_deficit[i] > 0 and col_deficit[j] > 0:
                X_int[i, j] += 1
                row_deficit[i] -= 1
                col_deficit[j] -= 1
                total_to_add -= 1
                progress = True
                if total_to_add == 0:
                    break

        if not progress:
            raise RuntimeError(
                "Could not complete integer rounding while preserving margins."
            )

    if not np.all(X_int.sum(axis=1) == row_sums):
        raise RuntimeError("Rounded matrix does not preserve row sums.")
    if not np.all(X_int.sum(axis=0) == col_sums):
        raise RuntimeError("Rounded matrix does not preserve column sums.")

    return X_int


def partition_dataset_constrained_dirichlet(
    df: pd.DataFrame,
    n_clients: int,
    beta_L: float,
    beta_Q: float,
    label_column: str = "Risk_Category_Composite",
    client_column: str = "Client",
    min_size: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Partition a dataset into clients using a global constrained Dirichlet approach.

    Main idea:
      1) sample exact client sizes from quantity skew
      2) sample client-specific label preferences from label skew
      3) use IPF to construct a global allocation matrix with:
           - exact client sizes (row sums)
           - exact global label counts (column sums)
      4) assign actual rows accordingly

    This avoids order effects and preserves both the quantity skew and
    the global label availability exactly.
    """
    rng = np.random.default_rng(seed)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe.")
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")
    if n_clients <= 0:
        raise ValueError("n_clients must be positive.")
    if beta_L <= 0:
        raise ValueError("beta_L must be positive.")
    if beta_Q <= 0:
        raise ValueError("beta_Q must be positive.")

    df = df.copy().reset_index(drop=True)
    n_total = len(df)

    # Labels
    labels = sorted(df[label_column].unique().tolist())
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # ------------------------------------------------------------
    # STEP 1: QUANTITY SKEW -> exact client sizes
    # ------------------------------------------------------------
    client_sizes = _sample_client_sizes(
        n_total=n_total,
        n_clients=n_clients,
        beta_Q=beta_Q,
        min_size=min_size,
        rng=rng,
    )

    # ------------------------------------------------------------
    # STEP 2: GLOBAL LABEL COUNTS
    # ------------------------------------------------------------
    global_counts = (
        df[label_column]
        .value_counts()
        .reindex(labels, fill_value=0)
        .to_numpy(dtype=int)
    )
    global_dist = global_counts / global_counts.sum()

    # ------------------------------------------------------------
    # STEP 3: LABEL SKEW -> client-specific preferences
    # ------------------------------------------------------------
    # Anchor each client's Dirichlet around the global distribution
    alpha = np.maximum(beta_L * global_dist, 1e-12)

    client_label_dists = np.array([
        rng.dirichlet(alpha) for _ in range(n_clients)
    ])

    # Seed matrix: what each client "wants" before global correction
    seed_matrix = _build_seed_matrix(
        client_sizes=client_sizes,
        client_label_dists=client_label_dists,
    )

    # ------------------------------------------------------------
    # STEP 4: GLOBAL CORRECTION VIA IPF
    # ------------------------------------------------------------
    allocation_real = _ipf(
        seed_matrix=seed_matrix,
        row_sums=client_sizes,
        col_sums=global_counts,
    )

    # ------------------------------------------------------------
    # STEP 5: INTEGER ROUNDING WITH EXACT MARGINS
    # ------------------------------------------------------------
    allocation_int = _round_matrix_preserving_margins(
        X=allocation_real,
        row_sums=client_sizes,
        col_sums=global_counts,
    )

    # ------------------------------------------------------------
    # STEP 6: ASSIGN ACTUAL ROWS BASED ON INTEGER ALLOCATION
    # ------------------------------------------------------------
    label_pools: Dict[int, List[int]] = {}
    for label in labels:
        idxs = df.index[df[label_column] == label].to_list()
        rng.shuffle(idxs)
        label_pools[label] = idxs

    client_assignments: Dict[int, List[int]] = {i: [] for i in range(n_clients)}

    for label in labels:
        j = label_to_idx[label]
        pool = label_pools[label]
        start = 0

        for client_id in range(n_clients):
            take = int(allocation_int[client_id, j])
            if take > 0:
                chosen = pool[start:start + take]
                if len(chosen) != take:
                    raise RuntimeError(
                        f"Not enough rows available for label {label}. "
                        f"Expected {take}, got {len(chosen)}."
                    )
                client_assignments[client_id].extend(chosen)
                start += take

        if start != len(pool):
            raise RuntimeError(
                f"Allocation for label {label} did not use all rows. "
                f"Used {start}, available {len(pool)}."
            )

    # ------------------------------------------------------------
    # FINAL CHECKS
    # ------------------------------------------------------------
    final_sizes = np.array([len(client_assignments[i]) for i in range(n_clients)])
    if not np.all(final_sizes == client_sizes):
        raise RuntimeError(
            f"Allocation failed. Expected client sizes {client_sizes}, got {final_sizes}."
        )

    all_assigned = [idx for idxs in client_assignments.values() for idx in idxs]
    if len(all_assigned) != n_total:
        raise RuntimeError(
            f"Not all rows were assigned exactly once. Assigned {len(all_assigned)} "
            f"rows, but dataset has {n_total} rows."
        )

    if len(set(all_assigned)) != n_total:
        raise RuntimeError("Some rows were assigned more than once.")

    # Write client ids back
    df[client_column] = -1
    for client_id, idxs in client_assignments.items():
        df.loc[idxs, client_column] = client_id + 1

    if (df[client_column] == -1).any():
        raise RuntimeError("Some rows did not receive a client assignment.")

    return df


def print_partition_statistics(
    df: pd.DataFrame,
    label_column: str,
    client_column: str = "Client",
) -> None:
    print("\nPartition Statistics")
    counts = df.groupby(client_column)[label_column].value_counts().unstack(fill_value=0)
    print(counts)


def compute_partition_heterogeneity_metrics(
    df: pd.DataFrame,
    label_column: str,
    client_column: str = "Client",
) -> dict:
    label_dist = (
        df.groupby(client_column)[label_column]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    global_dist = (
        df[label_column]
        .value_counts(normalize=True)
        .reindex(label_dist.columns, fill_value=0)
    )

    kl_list = []

    for _, row in label_dist.iterrows():
        p = row.to_numpy(dtype=float) + 1e-12
        q = global_dist.to_numpy(dtype=float) + 1e-12
        kl = np.sum(p * np.log(p / q))
        kl_list.append(kl)

    return {
        "mean_kl_divergence": float(np.mean(kl_list)),
        "std_kl_divergence": float(np.std(kl_list)),
    }


def print_quantity_skew_statistics(
    df: pd.DataFrame,
    client_column: str = "Client",
) -> None:
    print("\nQuantity Skew Statistics")
    counts = df[client_column].value_counts().sort_index()
    print(counts)


def compute_quantity_skew_metrics(
    df: pd.DataFrame,
    client_column: str = "Client",
) -> dict:
    counts = df[client_column].value_counts().values
    return {
        "min_client_size": int(np.min(counts)),
        "max_client_size": int(np.max(counts)),
        "std_client_size": float(np.std(counts)),
    }