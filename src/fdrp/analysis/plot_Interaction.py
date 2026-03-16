import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_interaction_heatmaps(
    csv_path,
    metric="f1_global_macro",
    beta_l_values=None,
    beta_q_values=None,
    paradigms=("centralized", "local", "federated"),
    figsize=(18, 5),
    save_path=None,
):
    """
    Plot one heatmap per paradigm over the beta_L x beta_Q grid.

    Parameters
    ----------
    csv_path : str
        Path to summary CSV.
    metric : str
        Metric column to visualize.
    beta_l_values : list[float] or None
        Ordered beta_L values for rows. If None, inferred from data.
    beta_q_values : list[float] or None
        Ordered beta_Q values for columns. If None, inferred from data.
    paradigms : tuple[str]
        Paradigms to include.
    figsize : tuple
        Figure size.
    save_path : str or None
        Optional save path.
    """

    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV.")

    if beta_l_values is None:
        beta_l_values = sorted(df["beta_L"].dropna().unique())

    if beta_q_values is None:
        beta_q_values = sorted(df["beta_Q"].dropna().unique())

    # Average over seeds if multiple seeds exist
    grouped = (
        df.groupby(["paradigm", "beta_L", "beta_Q"], as_index=False)[metric]
        .mean()
    )

    metric_labels = {
        "f1_global_macro": "F1 global macro",
        "f1_per_client_macro": "F1 per-client macro",
        "mse_macro": "MSE macro",
        "ece_macro": "ECE macro",
        "mae_macro": "MAE macro",
        "ece_prob_macro": "ECE probability macro",
        "fleiss_kappa_global_macro": "Fleiss kappa global macro",
        "fleiss_kappa_per_client_macro": "Fleiss kappa per-client macro",
        "disagreement_global_macro": "Disagreement global macro",
        "disagreement_per_client_macro": "Disagreement per-client macro",
    }
    pretty_metric = metric_labels.get(metric, metric)

    # Use same color scale across paradigms
    vmin = grouped[metric].min()
    vmax = grouped[metric].max()

    fig, axes = plt.subplots(1, len(paradigms), figsize=figsize, constrained_layout=True)

    if len(paradigms) == 1:
        axes = [axes]

    im = None

    for ax, paradigm in zip(axes, paradigms):
        sub = grouped[grouped["paradigm"] == paradigm].copy()

        pivot = (
            sub.pivot(index="beta_L", columns="beta_Q", values=metric)
            .reindex(index=beta_l_values, columns=beta_q_values)
        )

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(paradigm.capitalize(), fontsize=14)
        ax.set_xlabel(r"$\beta_Q$")
        ax.set_xticks(range(len(beta_q_values)))
        ax.set_xticklabels(beta_q_values)

        ax.set_ylabel(r"$\beta_L$")
        ax.set_yticks(range(len(beta_l_values)))
        ax.set_yticklabels(beta_l_values)

        # write values in cells
        for i in range(len(beta_l_values)):
            for j in range(len(beta_q_values)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95)
    cbar.set_label(pretty_metric)

    fig.suptitle(f"{pretty_metric} across $\\beta_L \\times \\beta_Q$", fontsize=16)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

BETA_L_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0]
BETA_Q_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0]
Oskar_path = r"C:\Users\oskar\OneDrive\Desktop\4 Semester\Dataproject\Federated-dental-risk-vol2\federated-dental-risk-prediction\src\fdrp\analysis\Data\sweep_beta_summary_1.csv"
plot_interaction_heatmaps(
    csv_path=Oskar_path,
    metric="f1_global_macro",
    beta_l_values=BETA_L_VALUES,
    beta_q_values=BETA_Q_VALUES,
)