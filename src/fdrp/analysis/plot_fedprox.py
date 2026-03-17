import pandas as pd
import matplotlib.pyplot as plt


def plot_fedprox_mu_sweep(
    csv_path,
    metric="f1_global_macro",
    seed=None,
    beta_L=None,
    beta_Q=None,
    figsize=(10, 6),
    save_path=None,
):

    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found.")

    if beta_L is not None:
        df = df[df["beta_L"] == beta_L]

    if beta_Q is not None:
        df = df[df["beta_Q"] == beta_Q]

    if seed is not None:
        df = df[df["seed"] == seed]

    metric_labels = {
        "f1_global_macro": "F1 global macro",
        "f1_per_client_macro": "F1 per-client macro",
        "mse_macro": "MSE macro",
        "ece_macro": "ECE macro",
        "mae_macro": "MAE macro",
        "ece_prob_macro": "ECE probability macro",
        "fleiss_kappa_global_macro": "Fleiss kappa global macro",
        "fleiss_kappa_per_client_macro": "Fleiss kappa per-client macro",
    }

    pretty_metric = metric_labels.get(metric, metric)

    grouped = (
        df.groupby(["paradigm", "federated_method", "fedprox_mu"], dropna=False, as_index=False)[metric]
        .mean()
    )

    centralized_value = grouped.loc[grouped["paradigm"] == "centralized", metric].mean()
    local_value = grouped.loc[grouped["paradigm"] == "local", metric].mean()

    federated_df = grouped[grouped["paradigm"] == "federated"]

    fedavg_df = federated_df[federated_df["fedprox_mu"] == 0.0]
    fedprox_df = federated_df[federated_df["fedprox_mu"] > 0].sort_values("fedprox_mu")

    plt.figure(figsize=figsize)

    # FedProx curve
    if not fedprox_df.empty:
        plt.plot(
            fedprox_df["fedprox_mu"],
            fedprox_df[metric],
            marker="o",
            linewidth=2,
            color="green",
            label="FedProx",
        )
        plt.xscale("log")

    # Centralized line
    if pd.notna(centralized_value):
        plt.axhline(
            centralized_value,
            linestyle="--",
            color="blue",
            linewidth=2,
            label=f"Centralized ({centralized_value:.3f})",
        )

    # Local line
    if pd.notna(local_value):
        plt.axhline(
            local_value,
            linestyle="--",
            color="red",
            linewidth=2,
            label=f"Local ({local_value:.3f})",
        )

    # FedAvg line
    if not fedavg_df.empty:
        fedavg_value = fedavg_df[metric].mean()
        plt.axhline(
            fedavg_value,
            linestyle="--",
            color="orange",
            linewidth=2,
            label=f"Federated / FedAvg μ=0 ({fedavg_value:.3f})",
        )

    title = f"{pretty_metric} vs FedProx μ"
    subtitle = []

    if beta_L is not None:
        subtitle.append(rf"$\beta_L={beta_L}$")

    if beta_Q is not None:
        subtitle.append(rf"$\beta_Q={beta_Q}$")

    if seed is not None:
        subtitle.append(f"seed={seed}")

    if subtitle:
        title += " | " + ", ".join(subtitle)

    plt.title(title)
    plt.xlabel(r"FedProx $\mu$ (log scale)")
    plt.ylabel(pretty_metric)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


Oskar_path = r"C:\Users\Oskar\Desktop\fedprox_mu_sweep_summary.csv"
plot_fedprox_mu_sweep(
    csv_path=Oskar_path,
    metric="f1_global_macro",
    beta_L=1.0,
    beta_Q=5.0,
)
