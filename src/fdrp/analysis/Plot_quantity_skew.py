import pandas as pd
import matplotlib.pyplot as plt


def plot_quantity_skew(
    csv_path,
    metric="f1_global_macro",
    beta_q_values=None,
    beta_l_values=None,
    paradigms=("centralized", "local", "federated"),
):

    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV.")

    if beta_q_values is None:
        beta_q_values = sorted(df["beta_Q"].dropna().unique())

    if beta_l_values is None:
        beta_l_values = sorted(df["beta_L"].dropna().unique())

    grouped = (
        df.groupby(["beta_Q", "beta_L", "paradigm"], as_index=False)[metric]
        .mean()
    )

    beta_l_groups = [beta_l_values[:4], beta_l_values[4:8]]

    for fig_idx, beta_l_group in enumerate(beta_l_groups, start=1):

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        axes = axes.flatten()

        for i, beta_l in enumerate(beta_l_group):

            ax = axes[i]

            panel_df = grouped[grouped["beta_L"] == beta_l]

            for paradigm in paradigms:

                line_df = panel_df[panel_df["paradigm"] == paradigm].copy()

                line_df["beta_Q"] = pd.Categorical(
                    line_df["beta_Q"],
                    categories=beta_q_values,
                    ordered=True,
                )

                line_df = line_df.sort_values("beta_Q")

                ax.plot(
                    line_df["beta_Q"].astype(float),
                    line_df[metric],
                    marker="o",
                    label=paradigm,
                )

            ax.set_title(rf"$\beta_L = {beta_l}$", fontsize=18)
            ax.grid(True, alpha=0.3)

            if i % 2 == 0:
                ax.set_ylabel(metric, fontsize=14)

            if i >= 2:
                ax.set_xlabel(r"$\beta_Q$", fontsize=14)

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(paradigms),
            frameon=False,
            bbox_to_anchor=(0.5, 0.95),
        )

        fig.suptitle(
            f"{metric} vs $\\beta_Q$ for fixed $\\beta_L$ (Part {fig_idx})",
            fontsize=22,
            y=0.98,
        )

        fig.subplots_adjust(top=0.82, hspace=0.28, wspace=0.15)

        plt.show()


Oskar_path = r"C:\Users\oskar\OneDrive\Desktop\4 Semester\Dataproject\Federated-dental-risk-vol2\federated-dental-risk-prediction\src\fdrp\analysis\Data\sweep_beta_summary_1.csv"

plot_quantity_skew(
    csv_path=Oskar_path,
    metric="ece_macro",
)