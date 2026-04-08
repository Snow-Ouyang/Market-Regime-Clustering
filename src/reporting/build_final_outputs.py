from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

EXT_RESULTS = ROOT / "results" / "extensions" / "stress_aware_credit_4state"
BASELINE_CORE = ROOT / "results" / "core"
FINAL_RESULTS = ROOT / "results" / "final_model"
BASELINE_RESULTS = ROOT / "results" / "baseline_reference"

FIGURES_DIR = ROOT / "figures"
FINAL_FIGURES = FIGURES_DIR / "final_model"
BASELINE_FIGURES = FIGURES_DIR / "baseline_reference"

FINAL_MACRO_PATH = ROOT / "data_processed" / "final_macro_panel.csv"


def ensure_dirs() -> None:
    for path in [FINAL_RESULTS, BASELINE_RESULTS, FINAL_FIGURES, BASELINE_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_final_characteristics() -> pd.DataFrame:
    moments = pd.read_csv(EXT_RESULTS / "stress_aware_4state_variable_moments.csv")
    macro = pd.read_csv(FINAL_MACRO_PATH, parse_dates=["date"])[["date", "realized_vol"]]
    panel = pd.read_csv(EXT_RESULTS / "stress_aware_4state_regime_panel.csv", parse_dates=["date"])
    realized = (
        panel[["date", "regime_name"]]
        .merge(macro, on="date", how="left")
        .groupby("regime_name", as_index=False)["realized_vol"]
        .mean()
        .rename(columns={"realized_vol": "realized_vol_mean"})
    )

    means = (
        moments.pivot(index="regime_name", columns="variable", values="mean")
        .reset_index()
        .rename(
            columns={
                "growth_pc1": "growth_pc1_mean",
                "inflation_pc1": "inflation_pc1_mean",
                "gs10": "gs10_mean",
                "term_spread_10y_1y": "term_spread_10y_1y_mean",
                "credit_spread": "credit_spread_mean",
            }
        )
    )

    asset_perf = pd.read_csv(EXT_RESULTS / "stress_aware_4state_asset_performance.csv")
    asset_wide = (
        asset_perf.pivot(index="regime_name", columns="asset", values="time_weighted_annualized_return")
        .reset_index()
        .rename(
            columns={
                "spx": "spx_ann_return",
                "bond": "bond_ann_return",
                "oil": "oil_ann_return",
                "gold": "gold_ann_return",
            }
        )
    )

    out = means.merge(realized, on="regime_name", how="left").merge(asset_wide, on="regime_name", how="left")
    return out


def plot_final_profile_heatmap(summary: pd.DataFrame) -> None:
    vars_for_plot = [
        "growth_pc1_mean",
        "inflation_pc1_mean",
        "gs10_mean",
        "term_spread_10y_1y_mean",
        "credit_spread_mean",
        "realized_vol_mean",
    ]
    mat = summary.set_index("regime_name")[vars_for_plot].copy()
    color_mat = (mat - mat.mean()) / mat.std(ddof=0)

    plt.figure(figsize=(10, 4.8))
    sns.heatmap(
        color_mat,
        annot=mat.round(2),
        fmt="",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Column-standardized mean"},
    )
    plt.title("Preferred 4-State Model: Regime Characteristics")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FINAL_FIGURES / "final_model_profile_heatmap.png", dpi=180)
    plt.close()


def build_outputs() -> None:
    ensure_dirs()

    final_files = {
        EXT_RESULTS / "stress_aware_4state_regime_labels.csv": FINAL_RESULTS / "regime_labels.csv",
        EXT_RESULTS / "stress_aware_4state_regime_panel.csv": FINAL_RESULTS / "regime_panel.csv",
        EXT_RESULTS / "stress_aware_4state_variable_moments.csv": FINAL_RESULTS / "state_moments.csv",
        EXT_RESULTS / "stress_aware_4state_asset_performance.csv": FINAL_RESULTS / "asset_performance_by_regime.csv",
        EXT_RESULTS / "plots" / "stress_aware_4state_state_path.png": FINAL_FIGURES / "stress_aware_regime_path.png",
        EXT_RESULTS / "plots" / "stress_aware_4state_asset_macro_events.png": FINAL_FIGURES / "stress_aware_assets_by_regime.png",
    }
    for src, dst in final_files.items():
        copy(src, dst)

    baseline_files = {
        BASELINE_CORE / "jump_model_penalty_grid" / "jump_model_penalty_grid_summary.csv": BASELINE_RESULTS / "penalty_grid_summary.csv",
        BASELINE_CORE / "jump_model_time_stability" / "jump_model_time_stability_summary.csv": BASELINE_RESULTS / "time_stability_summary.csv",
        BASELINE_CORE / "jump_model_state_count_stability" / "jump_model_state_count_summary.csv": BASELINE_RESULTS / "state_count_summary.csv",
        BASELINE_CORE / "regime_interpretation" / "regime_labels.csv": BASELINE_RESULTS / "regime_labels.csv",
        BASELINE_CORE / "regime_interpretation" / "asset_performance_by_regime.csv": BASELINE_RESULTS / "asset_performance_by_regime.csv",
        FIGURES_DIR / "final_regime_path.png": BASELINE_FIGURES / "baseline_regime_path.png",
        FIGURES_DIR / "jump_model_penalty_grid_diagnostics.png": BASELINE_FIGURES / "baseline_penalty_diagnostics.png",
        FIGURES_DIR / "jump_model_time_stability_paths.png": BASELINE_FIGURES / "baseline_time_stability_paths.png",
        FIGURES_DIR / "jump_model_state_count_paths.png": BASELINE_FIGURES / "baseline_state_count_paths.png",
        FIGURES_DIR / "jump_model_variable_stability_paths.png": BASELINE_FIGURES / "baseline_variable_stability_paths.png",
        FIGURES_DIR / "regime_external_validation_heatmap.png": BASELINE_FIGURES / "baseline_external_validation_heatmap.png",
    }
    for src, dst in baseline_files.items():
        copy(src, dst)

    characteristics = build_final_characteristics()
    characteristics.to_csv(FINAL_RESULTS / "regime_characteristics_summary.csv", index=False)
    plot_final_profile_heatmap(characteristics)


def main() -> None:
    build_outputs()
    print("Final model outputs assembled under results/final_model and figures/final_model.")


if __name__ == "__main__":
    main()
