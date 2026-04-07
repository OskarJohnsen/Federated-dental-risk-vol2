import pandas as pd
import matplotlib.pyplot as plt


def plot_label_skew_in_two_figures(
    csv_path,
    metric="f1_global_macro",
    beta_q_values=None,
    beta_l_values=None,
    paradigms=("centralized", "local", "federated"),
    min_client_col="partition_min_client_size",
    max_client_col="partition_max_client_size",
    show_client_table=True,
    save_prefix=None,
):
    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV.")

    if beta_q_values is None:
        beta_q_values = sorted(df["beta_Q"].dropna().unique())

    if beta_l_values is None:
        beta_l_values = sorted(df["beta_L"].dropna().unique())

    metric_labels = {
        "f1_global_macro": "F1 global macro",
        "f1_per_client_macro": "F1 per-client macro",
        "mse_macro": "MSE macro",
        "ece_macro": "ECE macro",
        "fleiss_kappa_global_macro": "Fleiss kappa global macro",
        "fleiss_kappa_per_client_macro": "Fleiss kappa per-client macro",
    }
    pretty_metric = metric_labels.get(metric, metric)

    grouped_metric = (
        df.groupby(["beta_Q", "beta_L", "paradigm"], as_index=False)[metric]
        .mean()
    )

    has_client_info = (
        show_client_table
        and min_client_col in df.columns
        and max_client_col in df.columns
    )

    if has_client_info:
        grouped_client_info = (
            df.groupby(["beta_Q", "beta_L"], as_index=False)[[min_client_col, max_client_col]]
            .mean()
        )

    beta_q_groups = [beta_q_values[:4], beta_q_values[4:8]]

    for fig_idx, beta_q_group in enumerate(beta_q_groups, start=1):
        if not beta_q_group:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        axes = axes.flatten()

        for i, beta_q in enumerate(beta_q_group):
            ax = axes[i]

            panel_df = grouped_metric[grouped_metric["beta_Q"] == beta_q].copy()

            for paradigm in paradigms:
                line_df = panel_df[panel_df["paradigm"] == paradigm].copy()
                line_df["beta_L"] = pd.Categorical(
                    line_df["beta_L"],
                    categories=beta_l_values,
                    ordered=True,
                )
                line_df = line_df.sort_values("beta_L")

                ax.plot(
                    line_df["beta_L"].astype(float),
                    line_df[metric],
                    marker="o",
                    label=paradigm,
                )

            ax.set_title(rf"$\beta_Q = {beta_q}$", fontsize=18)
            ax.grid(True, alpha=0.3)

            # Kun ylabel på venstre kolonne
            if i % 2 == 0:
                ax.set_ylabel(pretty_metric, fontsize=14)

            # Kun xlabel på nederste række
            if i >= 2:
                ax.set_xlabel(r"$\beta_L$", fontsize=14)

            # Tabel
            if has_client_info:
                info_df = grouped_client_info[grouped_client_info["beta_Q"] == beta_q].copy()
                info_df["beta_L"] = pd.Categorical(
                    info_df["beta_L"],
                    categories=beta_l_values,
                    ordered=True,
                )
                info_df = info_df.sort_values("beta_L")

                min_client = int(round(info_df[min_client_col].min()))
                max_client = int(round(info_df[max_client_col].max()))

                table_data = [
                    ["Min client", f"{min_client}"],
                    ["Max client", f"{max_client}"],
                ]

                tbl = ax.table(
                    cellText=table_data,
                    colLabels=["Client info", "Size"],
                    cellLoc="center",
                    bbox=[0.60, 0.03, 0.36, 0.20],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

        for j in range(len(beta_q_group), 4):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(paradigms),
            frameon=False,
            bbox_to_anchor=(0.5, 0.95),
            fontsize=13,
        )

        fig.suptitle(
            f"{pretty_metric} vs $\\beta_L$ for fixed $\\beta_Q$ (Part {fig_idx})",
            fontsize=24,
            y=0.98,
        )

        fig.subplots_adjust(top=0.82, hspace=0.28, wspace=0.15)

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_part{fig_idx}.png", dpi=300, bbox_inches="tight")

        plt.show()


Oskar_path = r"C:\Users\Oskar\Desktop\Dataproject\federated-dental-risk-vol2\src\fdrp\analysis\Data\sweep_beta_summary_2.csv"


BETA_L_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0]
BETA_Q_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0]

plot_label_skew_in_two_figures(
    csv_path=Oskar_path,
    metric="ece_macro",
    beta_q_values=BETA_Q_VALUES,
    beta_l_values=BETA_L_VALUES,
)
