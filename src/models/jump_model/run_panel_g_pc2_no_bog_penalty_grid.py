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
RESULTS_DIR = ROOT / "results" / "core" / "jump_model_penalty_grid"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
]
N_STATES = 3
SEEDS = list(range(10))
PENALTIES = [0.50, 0.55, 0.60, 0.65, 0.70]


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
    # The original panel is retained for interpretation and stacked output.
    return StandardScaler().fit_transform(df[FEATURES])


def fit_one_jump_model(X: np.ndarray, penalty: float, seed: int) -> JumpModel:
    model = JumpModel(
        n_components=N_STATES,
        jump_penalty=penalty,
        cont=False,
        random_state=seed,
        max_iter=1000,
        tol=1e-8,
        n_init=10,
        verbose=0,
    )
    model.fit(X, sort_by="freq")
    return model


def select_best_model(X: np.ndarray, penalty: float) -> tuple[JumpModel, int, float]:
    best_model = None
    best_seed = None
    best_obj = np.inf
    for seed in SEEDS:
        model = fit_one_jump_model(X, penalty, seed)
        objective = float(model.val_)
        if objective < best_obj:
            best_obj = objective
            best_model = model
            best_seed = seed
    if best_model is None:
        raise RuntimeError(f"No successful Jump Model fit for penalty={penalty}")
    return best_model, int(best_seed), float(best_obj)


def build_state_panel(panel: pd.DataFrame, model: JumpModel, penalty: float, best_seed: int) -> pd.DataFrame:
    out = panel.copy()
    out["penalty"] = penalty
    out["best_seed"] = best_seed
    out["state"] = np.asarray(model.labels_).astype(int)
    proba = np.asarray(model.proba_)
    for i in range(N_STATES):
        out[f"state_prob_{i}"] = proba[:, i]
    return out


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    return (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )


def summarize_penalty(state_panel: pd.DataFrame, objective_value: float) -> tuple[dict[str, object], pd.DataFrame]:
    shares = state_panel["state"].value_counts(normalize=True).sort_index().reindex(range(N_STATES), fill_value=0.0)
    runs = extract_runs(state_panel["state"])

    duration_by_state = {}
    occ_by_state = {}
    for state in range(N_STATES):
        durations = runs.loc[runs["state"] == state, "duration"]
        duration_by_state[state] = float(durations.mean()) if len(durations) else np.nan
        occ_by_state[state] = int(len(durations))

    flags = []
    if float(shares.min()) < 0.05:
        flags.append("too_fragmented")
    if float(shares.max()) > 0.70:
        flags.append("too_collapsed")
    if float(np.nanmin(list(duration_by_state.values()))) < 3:
        flags.append("too_short")
    label = "candidate" if not flags else "|".join(flags)

    summary_row = {
        "penalty": float(state_panel["penalty"].iloc[0]),
        "best_seed": int(state_panel["best_seed"].iloc[0]),
        "objective_value": objective_value,
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
        "state_0_share": float(shares.loc[0]),
        "state_1_share": float(shares.loc[1]),
        "state_2_share": float(shares.loc[2]),
        "state_0_mean_duration": duration_by_state[0],
        "state_1_mean_duration": duration_by_state[1],
        "state_2_mean_duration": duration_by_state[2],
        "smallest_state_mean_duration": float(np.nanmin(list(duration_by_state.values()))),
        "largest_state_mean_duration": float(np.nanmax(list(duration_by_state.values()))),
        "avg_state_duration": float(np.nanmean(list(duration_by_state.values()))),
        "num_states_below_5pct_share": int((shares < 0.05).sum()),
        "num_states_below_3_month_mean_duration": int(sum(x < 3 for x in duration_by_state.values() if pd.notna(x))),
        "state_0_occurrences": occ_by_state[0],
        "state_1_occurrences": occ_by_state[1],
        "state_2_occurrences": occ_by_state[2],
        "screening_label": label,
    }
    return summary_row, runs


def state_profile_frame(stacked_panel: pd.DataFrame) -> pd.DataFrame:
    profile = (
        stacked_panel.groupby(["penalty", "state"])[FEATURES]
        .mean()
        .reset_index()
        .melt(id_vars=["penalty", "state"], value_vars=FEATURES, var_name="variable", value_name="mean_value")
    )
    return profile


def plot_multi_penalty_paths(stacked_panel: pd.DataFrame) -> None:
    penalties = sorted(stacked_panel["penalty"].unique())
    colors = sns.color_palette("tab10", N_STATES)
    fig, axes = plt.subplots(len(penalties), 1, figsize=(13, 2.2 * len(penalties)), sharex=True)
    if len(penalties) == 1:
        axes = [axes]
    for ax, penalty in zip(axes, penalties):
        subset = stacked_panel.loc[stacked_panel["penalty"] == penalty].copy()
        ax.step(subset["date"], subset["state"], where="post", color="#222222", linewidth=1.1)
        for state in range(N_STATES):
            mask = subset["state"] == state
            ax.fill_between(subset["date"], -0.5, N_STATES - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels([f"state_{i}" for i in range(N_STATES)])
        ax.set_ylabel(f"p={penalty:g}")
    axes[0].set_title("panel_g_pc2_no_bog: 3-State Jump Model Paths by Penalty")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_penalty_grid_paths.png", dpi=180)
    plt.close()


def plot_multi_penalty_profiles(profile_long: pd.DataFrame) -> None:
    penalties = sorted(profile_long["penalty"].unique())
    fig, axes = plt.subplots(len(penalties), 1, figsize=(10, 2.6 * len(penalties)), sharex=False)
    if len(penalties) == 1:
        axes = [axes]
    for ax, penalty in zip(axes, penalties):
        subset = profile_long.loc[profile_long["penalty"] == penalty]
        mat = subset.pivot(index="variable", columns="state", values="mean_value").reindex(FEATURES)
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar=ax is axes[0])
        ax.set_title(f"Penalty = {penalty:g}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_penalty_grid_profiles.png", dpi=180)
    plt.close()


def plot_penalty_diagnostics(summary: pd.DataFrame) -> None:
    metrics = [
        ("min_state_share", "Min State Share"),
        ("max_state_share", "Max State Share"),
        ("smallest_state_mean_duration", "Smallest Mean Duration"),
        ("largest_state_mean_duration", "Largest Mean Duration"),
        ("avg_state_duration", "Average Mean Duration"),
        ("objective_value", "Objective Value"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(summary["penalty"], summary[metric], marker="o", color="#1f3b73", linewidth=1.5)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel("penalty")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jump_model_penalty_grid_diagnostics.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    panel = load_panel()
    stacked_rows = []
    summary_rows = []

    for penalty in PENALTIES:
        X = zscore_features(panel)
        model, best_seed, objective_value = select_best_model(X, penalty)
        state_panel = build_state_panel(panel, model, penalty, best_seed)
        summary_row, _ = summarize_penalty(state_panel, objective_value)
        stacked_rows.append(state_panel)
        summary_rows.append(summary_row)

    stacked_panel = pd.concat(stacked_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("penalty").reset_index(drop=True)
    profile_long = state_profile_frame(stacked_panel)

    stacked_panel.to_csv(RESULTS_DIR / "jump_model_penalty_grid_panel.csv", index=False)
    summary.to_csv(RESULTS_DIR / "jump_model_penalty_grid_summary.csv", index=False)

    plot_multi_penalty_paths(stacked_panel)
    plot_multi_penalty_profiles(profile_long)
    plot_penalty_diagnostics(summary)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
