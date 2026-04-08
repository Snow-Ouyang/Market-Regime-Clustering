from __future__ import annotations

from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[3] / ".vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
RESULTS_DIR = ROOT / "results" / "core" / "jump_model_time_stability"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
]
PROFILE_VARS = FEATURES.copy()
N_STATES = 3
PENALTY = 0.6
SEEDS = list(range(10))


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> None:
    for pattern in ["*.csv", "*.png"]:
        for path in RESULTS_DIR.glob(pattern):
            path.unlink()
    for path in PLOTS_DIR.glob("*.png"):
        path.unlink()


def load_panel() -> pd.DataFrame:
    panel = (
        pd.read_csv(INPUT_PATH, parse_dates=["date"])
        .sort_values("date")
        .loc[:, ["date"] + FEATURES]
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )
    return panel


def build_sample_versions(panel: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    start = panel["date"].min()
    end = panel["date"].max()

    samples: list[tuple[str, pd.DataFrame]] = [("baseline", panel.copy())]
    for years in [3, 5, 10]:
        samples.append((f"trim_start_{years}y", panel.loc[panel["date"] >= start + pd.DateOffset(years=years)].copy()))
    for years in [3, 5, 10]:
        samples.append((f"trim_end_{years}y", panel.loc[panel["date"] <= end - pd.DateOffset(years=years)].copy()))

    both = panel.loc[
        (panel["date"] >= start + pd.DateOffset(years=5))
        & (panel["date"] <= end - pd.DateOffset(years=5))
    ].copy()
    samples.append(("trim_start_5y_end_5y", both))

    return [(name, sample.reset_index(drop=True)) for name, sample in samples if len(sample) > 60]


def zscore_features(df: pd.DataFrame) -> np.ndarray:
    # Jump Model fitting uses the standardized feature matrix only.
    # The original-scale panel is preserved for state interpretation and summaries.
    return StandardScaler().fit_transform(df[FEATURES])


def fit_one_jump_model(X: np.ndarray, seed: int) -> JumpModel:
    model = JumpModel(
        n_components=N_STATES,
        jump_penalty=PENALTY,
        cont=False,
        random_state=seed,
        max_iter=1000,
        tol=1e-8,
        n_init=10,
        verbose=0,
    )
    model.fit(X, sort_by="freq")
    return model


def select_best_model(X: np.ndarray) -> tuple[JumpModel, int, float]:
    best_model = None
    best_seed = None
    best_obj = np.inf
    for seed in SEEDS:
        model = fit_one_jump_model(X, seed)
        objective = float(model.val_)
        if objective < best_obj:
            best_obj = objective
            best_model = model
            best_seed = seed
    if best_model is None:
        raise RuntimeError("No successful Jump Model fit.")
    return best_model, int(best_seed), float(best_obj)


def build_state_panel(sample_name: str, panel: pd.DataFrame, model: JumpModel, best_seed: int) -> pd.DataFrame:
    out = panel.copy()
    out["sample_name"] = sample_name
    out["penalty"] = PENALTY
    out["best_seed"] = best_seed
    out["state"] = np.asarray(model.labels_).astype(int)
    proba = np.asarray(model.proba_)
    for i in range(N_STATES):
        out[f"state_prob_{i}"] = proba[:, i]
    cols = ["sample_name", "date", "penalty", "best_seed", "state"] + [f"state_prob_{i}" for i in range(N_STATES)] + FEATURES
    return out[cols]


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    return (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )


def summarize_one_sample(state_panel: pd.DataFrame, objective_value: float) -> tuple[dict[str, object], pd.DataFrame]:
    shares = state_panel["state"].value_counts(normalize=True).sort_index().reindex(range(N_STATES), fill_value=0.0)
    counts = state_panel["state"].value_counts().sort_index().reindex(range(N_STATES), fill_value=0)
    runs = extract_runs(state_panel["state"])

    duration_by_state: dict[int, float] = {}
    occ_by_state: dict[int, int] = {}
    for state in range(N_STATES):
        d = runs.loc[runs["state"] == state, "duration"]
        duration_by_state[state] = float(d.mean()) if len(d) else np.nan
        occ_by_state[state] = int(len(d))

    summary = {
        "sample_name": state_panel["sample_name"].iloc[0],
        "penalty": PENALTY,
        "best_seed": int(state_panel["best_seed"].iloc[0]),
        "objective_value": objective_value,
        "start_date": state_panel["date"].min().strftime("%Y-%m-%d"),
        "end_date": state_panel["date"].max().strftime("%Y-%m-%d"),
        "n_obs": int(len(state_panel)),
        "state_0_share": float(shares.loc[0]),
        "state_1_share": float(shares.loc[1]),
        "state_2_share": float(shares.loc[2]),
        "state_0_mean_duration": duration_by_state[0],
        "state_1_mean_duration": duration_by_state[1],
        "state_2_mean_duration": duration_by_state[2],
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
        "smallest_state_mean_duration": float(np.nanmin(list(duration_by_state.values()))),
        "largest_state_mean_duration": float(np.nanmax(list(duration_by_state.values()))),
        "state_0_occurrences": occ_by_state[0],
        "state_1_occurrences": occ_by_state[1],
        "state_2_occurrences": occ_by_state[2],
        "state_0_count": int(counts.loc[0]),
        "state_1_count": int(counts.loc[1]),
        "state_2_count": int(counts.loc[2]),
    }

    profile = (
        state_panel.groupby("state")[PROFILE_VARS]
        .agg(["mean", "std", "count"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "variable"})
    )
    profile["sample_name"] = state_panel["sample_name"].iloc[0]
    profile["state"] = profile["state"].map(lambda x: f"state_{int(x)}")
    profile = profile[["sample_name", "state", "variable", "mean", "std", "count"]]

    return summary, profile


def plot_state_paths(stacked_panel: pd.DataFrame) -> None:
    order = [
        "baseline",
        "trim_start_3y",
        "trim_start_5y",
        "trim_start_10y",
        "trim_end_3y",
        "trim_end_5y",
        "trim_end_10y",
        "trim_start_5y_end_5y",
    ]
    samples = [x for x in order if x in stacked_panel["sample_name"].unique()]
    colors = sns.color_palette("tab10", N_STATES)
    fig, axes = plt.subplots(len(samples), 1, figsize=(13, 2.25 * len(samples)), sharex=True)
    if len(samples) == 1:
        axes = [axes]
    x_min = stacked_panel["date"].min()
    x_max = stacked_panel["date"].max()
    for ax, sample_name in zip(axes, samples):
        subset = stacked_panel.loc[stacked_panel["sample_name"] == sample_name].copy()
        ax.step(subset["date"], subset["state"], where="post", color="#222222", linewidth=1.1)
        for state in range(N_STATES):
            mask = subset["state"] == state
            ax.fill_between(subset["date"], -0.5, N_STATES - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels([f"state_{i}" for i in range(N_STATES)])
        ax.set_ylabel(sample_name)
    axes[0].set_title("Jump Model Time Stability: State Paths Across Trimmed Samples")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_time_stability_paths.png", dpi=180)
    plt.close()


def plot_share_duration_summary(summary: pd.DataFrame) -> None:
    metrics = [
        ("min_state_share", "Min State Share"),
        ("max_state_share", "Max State Share"),
        ("smallest_state_mean_duration", "Smallest Mean Duration"),
        ("largest_state_mean_duration", "Largest Mean Duration"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(summary["sample_name"], summary[metric], color="#3f8fc5")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_time_stability_summary_metrics.png", dpi=180)
    plt.close()


def plot_profile_heatmap(profile: pd.DataFrame) -> None:
    profile = profile.copy()
    profile["sample_state"] = profile["sample_name"] + "_" + profile["state"]
    mat = profile.pivot(index="sample_state", columns="variable", values="mean").reindex(columns=PROFILE_VARS)
    plt.figure(figsize=(10, 0.4 * len(mat) + 2.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Jump Model Time Stability: State Mean Profiles")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_time_stability_state_profile_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    full_panel = load_panel()
    samples = build_sample_versions(full_panel)

    panel_rows = []
    summary_rows = []
    profile_rows = []
    for sample_name, sample in samples:
        X = zscore_features(sample)
        model, best_seed, objective_value = select_best_model(X)
        state_panel = build_state_panel(sample_name, sample, model, best_seed)
        summary, profile = summarize_one_sample(state_panel, objective_value)
        panel_rows.append(state_panel)
        summary_rows.append(summary)
        profile_rows.append(profile)

    stacked_panel = pd.concat(panel_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    profile_df = pd.concat(profile_rows, ignore_index=True)

    stacked_panel.to_csv(RESULTS_DIR / "jump_model_time_stability_panel.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "jump_model_time_stability_summary.csv", index=False)
    profile_df.to_csv(RESULTS_DIR / "jump_model_time_stability_state_profile.csv", index=False)

    plot_state_paths(stacked_panel)
    plot_share_duration_summary(summary_df)
    plot_profile_heatmap(profile_df)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
