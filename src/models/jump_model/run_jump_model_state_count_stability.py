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
RESULTS_DIR = ROOT / "results" / "core" / "jump_model_state_count_stability"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
]
STATE_COUNTS = [2, 3, 4]
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
    return (
        pd.read_csv(INPUT_PATH, parse_dates=["date"])
        .sort_values("date")
        .loc[:, ["date"] + FEATURES]
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )


def zscore_features(df: pd.DataFrame) -> np.ndarray:
    # Jump Model fitting uses the standardized feature matrix only.
    # The original-scale panel is preserved for interpretation and summaries.
    return StandardScaler().fit_transform(df[FEATURES])


def fit_one_jump_model(X: np.ndarray, n_states: int, seed: int) -> JumpModel:
    model = JumpModel(
        n_components=n_states,
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


def select_best_model(X: np.ndarray, n_states: int) -> tuple[JumpModel, int, float]:
    best_model = None
    best_seed = None
    best_obj = np.inf
    for seed in SEEDS:
        model = fit_one_jump_model(X, n_states, seed)
        objective = float(model.val_)
        if objective < best_obj:
            best_obj = objective
            best_model = model
            best_seed = seed
    if best_model is None:
        raise RuntimeError(f"No successful Jump Model fit for n_states={n_states}")
    return best_model, int(best_seed), float(best_obj)


def build_state_panel(panel: pd.DataFrame, model: JumpModel, n_states: int, best_seed: int) -> pd.DataFrame:
    out = panel.copy()
    out["n_states"] = n_states
    out["penalty"] = PENALTY
    out["best_seed"] = best_seed
    out["state"] = np.asarray(model.labels_).astype(int)
    proba = np.asarray(model.proba_)
    for i in range(n_states):
        out[f"state_prob_{i}"] = proba[:, i]
    cols = ["n_states", "penalty", "best_seed", "date", "state"] + [f"state_prob_{i}" for i in range(n_states)] + FEATURES
    return out[cols]


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    return (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )


def summarize_one_model(state_panel: pd.DataFrame, n_states: int, objective_value: float) -> tuple[dict[str, object], pd.DataFrame]:
    shares = state_panel["state"].value_counts(normalize=True).sort_index().reindex(range(n_states), fill_value=0.0)
    runs = extract_runs(state_panel["state"])
    profile = (
        state_panel.groupby("state")[FEATURES]
        .agg(["mean", "std", "count"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "variable"})
    )
    profile["n_states"] = n_states
    profile["state"] = profile["state"].map(lambda x: f"state_{int(x)}")
    profile = profile[["n_states", "state", "variable", "mean", "std", "count"]]

    row: dict[str, object] = {
        "n_states": n_states,
        "best_seed": int(state_panel["best_seed"].iloc[0]),
        "objective_value": objective_value,
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
    }

    mean_durations: list[float] = []
    for state in range(n_states):
        row[f"state_{state}_share"] = float(shares.loc[state])
        d = runs.loc[runs["state"] == state, "duration"]
        mean_duration = float(d.mean()) if len(d) else np.nan
        row[f"state_{state}_mean_duration"] = mean_duration
        row[f"state_{state}_occurrences"] = int(len(d))
        mean_durations.append(mean_duration)

    row["smallest_state_mean_duration"] = float(np.nanmin(mean_durations))
    row["largest_state_mean_duration"] = float(np.nanmax(mean_durations))
    row["avg_state_duration"] = float(np.nanmean(mean_durations))
    row["num_states_below_5pct_share"] = int((shares < 0.05).sum())
    row["num_states_below_3_month_mean_duration"] = int(sum(x < 3 for x in mean_durations if pd.notna(x)))

    return row, profile


def plot_state_paths(stacked_panel: pd.DataFrame) -> None:
    colors = sns.color_palette("tab10", 4)
    fig, axes = plt.subplots(len(STATE_COUNTS), 1, figsize=(13, 2.4 * len(STATE_COUNTS)), sharex=True)
    if len(STATE_COUNTS) == 1:
        axes = [axes]
    for ax, n_states in zip(axes, STATE_COUNTS):
        subset = stacked_panel.loc[stacked_panel["n_states"] == n_states].copy()
        ax.step(subset["date"], subset["state"], where="post", color="#222222", linewidth=1.1)
        for state in range(n_states):
            mask = subset["state"] == state
            ax.fill_between(subset["date"], -0.5, n_states - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
        ax.set_yticks(range(n_states))
        ax.set_yticklabels([f"state_{i}" for i in range(n_states)])
        ax.set_ylabel(f"{n_states}-state")
    axes[0].set_title("Jump Model State Count Stability: State Paths")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_state_count_paths.png", dpi=180)
    plt.close()


def plot_summary_metrics(summary: pd.DataFrame) -> None:
    metrics = [
        ("min_state_share", "Min State Share"),
        ("max_state_share", "Max State Share"),
        ("smallest_state_mean_duration", "Smallest Mean Duration"),
        ("largest_state_mean_duration", "Largest Mean Duration"),
        ("avg_state_duration", "Average Mean Duration"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(summary["n_states"].astype(str), summary[metric], color="#3f8fc5")
        ax.set_title(title)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_state_count_summary_metrics.png", dpi=180)
    plt.close()


def plot_profile_heatmap(profile: pd.DataFrame) -> None:
    profile = profile.copy()
    profile["row_name"] = profile["n_states"].astype(str) + "_" + profile["state"]
    mat = profile.pivot(index="row_name", columns="variable", values="mean").reindex(columns=FEATURES)
    plt.figure(figsize=(10, 0.45 * len(mat) + 2.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Jump Model State Count Stability: State Mean Profiles")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_state_count_profile_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    panel = load_panel()
    X = zscore_features(panel)

    panels = []
    summaries = []
    profiles = []
    for n_states in STATE_COUNTS:
        model, best_seed, objective_value = select_best_model(X, n_states)
        state_panel = build_state_panel(panel, model, n_states, best_seed)
        summary_row, profile = summarize_one_model(state_panel, n_states, objective_value)
        panels.append(state_panel)
        summaries.append(summary_row)
        profiles.append(profile)

    stacked_panel = pd.concat(panels, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("n_states").reset_index(drop=True)
    profile_df = pd.concat(profiles, ignore_index=True)

    stacked_panel.to_csv(RESULTS_DIR / "jump_model_state_count_panel.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "jump_model_state_count_summary.csv", index=False)
    profile_df.to_csv(RESULTS_DIR / "jump_model_state_count_profile.csv", index=False)

    plot_state_paths(stacked_panel)
    plot_summary_metrics(summary_df)
    plot_profile_heatmap(profile_df)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
