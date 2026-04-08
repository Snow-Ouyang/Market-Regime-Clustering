from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "core" / "jump_model_penalty_grid"
INPUT_PATH = RESULTS_DIR / "jump_model_penalty_grid_panel.csv"
TARGET_PENALTY = 0.6

PROFILE_VARS = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
]


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values(["penalty", "date"]).reset_index(drop=True)
    panel = panel.loc[np.isclose(panel["penalty"], TARGET_PENALTY)].reset_index(drop=True)
    return panel


def build_long_profile(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        panel.groupby(["penalty", "state"])[PROFILE_VARS]
        .agg(["mean", "std", "count"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_2": "variable"})
    )

    shares = (
        panel.groupby(["penalty", "state"])
        .size()
        .rename("count_rows")
        .reset_index()
    )
    totals = shares.groupby("penalty")["count_rows"].transform("sum")
    shares["state_share"] = shares["count_rows"] / totals

    out = grouped.merge(shares[["penalty", "state", "state_share"]], on=["penalty", "state"], how="left")
    out["state"] = out["state"].map(lambda x: f"state_{int(x)}")
    out = out[["penalty", "state", "variable", "mean", "std", "count", "state_share"]]
    return out.sort_values(["penalty", "state", "variable"]).reset_index(drop=True)


def build_wide_profile(long_profile: pd.DataFrame) -> pd.DataFrame:
    base = (
        long_profile.pivot_table(
            index=["penalty", "state"],
            columns="variable",
            values=["mean", "std", "count"],
            aggfunc="first",
        )
        .sort_index(axis=1, level=[1, 0])
    )
    base.columns = [f"{var}_{stat}" for stat, var in base.columns]

    shares = (
        long_profile[["penalty", "state", "state_share"]]
        .drop_duplicates()
        .set_index(["penalty", "state"])
    )
    out = shares.join(base).reset_index()
    return out.sort_values(["penalty", "state"]).reset_index(drop=True)


def build_means_only(long_profile: pd.DataFrame) -> pd.DataFrame:
    out = (
        long_profile.pivot_table(
            index=["penalty", "state"],
            columns="variable",
            values="mean",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["penalty", "state"])
        .reset_index(drop=True)
    )
    return out


def plot_means_heatmap(means_only: pd.DataFrame) -> None:
    plot_df = means_only.copy()
    plot_df["penalty_state"] = plot_df.apply(lambda x: f"p{x['penalty']:g}_{x['state']}", axis=1)
    mat = plot_df.set_index("penalty_state")[PROFILE_VARS]

    plt.figure(figsize=(10, 0.45 * len(mat) + 2.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Jump Model Penalty-State Mean Profiles")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "jump_model_penalty_state_means_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    panel = load_panel()

    # Keep original model state labels. Later economic interpretation may require
    # manual relabeling across penalties if the state ordering differs.
    long_profile = build_long_profile(panel)
    wide_profile = build_wide_profile(long_profile)
    means_only = build_means_only(long_profile)

    long_profile.to_csv(RESULTS_DIR / "jump_model_penalty_state_profile_long.csv", index=False)
    wide_profile.to_csv(RESULTS_DIR / "jump_model_penalty_state_profile_wide.csv", index=False)
    means_only.to_csv(RESULTS_DIR / "jump_model_penalty_state_means_only.csv", index=False)
    plot_means_heatmap(means_only)

    print("Saved penalty-state profile summaries.")


if __name__ == "__main__":
    main()
