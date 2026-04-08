from __future__ import annotations

from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[2] / ".vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
RESULTS_DIR = ROOT / "results" / "appendix" / "archive_jump_model_full"
PLOTS_DIR = RESULTS_DIR / "plots"
STATE_COUNTS = [2, 3, 4]
SEEDS = list(range(10))
PENALTIES = [10.0, 25.0, 50.0, 100.0]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_panel() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    panel = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in panel.columns if c != "date"]
    panel = panel.dropna(subset=feature_cols).reset_index(drop=True)

    # Jump Model is fit on the standardized feature matrix only.
    # The original panel is preserved for interpretation, summaries, and plots.
    X = StandardScaler().fit_transform(panel[feature_cols])
    return panel, X, feature_cols


def fit_one_jump_model(X: np.ndarray, n_states: int, penalty: float, seed: int) -> JumpModel:
    # Use the discrete Jump Model (cont=False) because the task is regime classification
    # into distinct states rather than continuous state mixtures.
    model = JumpModel(
        n_components=n_states,
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


def select_best_model(X: np.ndarray, n_states: int) -> tuple[JumpModel, pd.DataFrame]:
    rows = []
    best_model = None
    best_obj = np.inf
    best_seed = None
    best_penalty = None

    for penalty in PENALTIES:
        for seed in SEEDS:
            try:
                model = fit_one_jump_model(X, n_states, penalty, seed)
                objective = float(model.val_)
                rows.append(
                    {
                        "n_states": n_states,
                        "penalty": penalty,
                        "seed": seed,
                        "objective_value": objective,
                        "completed": True,
                    }
                )
                if objective < best_obj:
                    best_obj = objective
                    best_model = model
                    best_seed = seed
                    best_penalty = penalty
            except Exception as exc:
                rows.append(
                    {
                        "n_states": n_states,
                        "penalty": penalty,
                        "seed": seed,
                        "objective_value": np.nan,
                        "completed": False,
                        "error": str(exc),
                    }
                )

    if best_model is None:
        raise RuntimeError(f"No successful Jump Model fit for n_states={n_states}")

    search_df = pd.DataFrame(rows)
    search_df["best_seed"] = best_seed
    search_df["best_penalty"] = best_penalty
    return best_model, search_df


def build_state_panel(panel: pd.DataFrame, model: JumpModel, n_states: int) -> pd.DataFrame:
    out = panel.copy()
    states = np.asarray(model.labels_).astype(int)
    out["state"] = states

    proba = np.asarray(model.proba_)
    for i in range(n_states):
        out[f"state_prob_{i}"] = proba[:, i]
    return out


def summarize_states(state_panel: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    total = len(state_panel)
    for state, group in state_panel.groupby("state"):
        share = len(group) / total
        for col in value_cols:
            rows.append(
                {
                    "state": f"state_{state}",
                    "variable": col,
                    "mean": group[col].mean(),
                    "std": group[col].std(),
                    "count": group[col].count(),
                    "sample_share": share,
                }
            )
    return pd.DataFrame(rows)


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    runs = (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )
    return runs


def summarize_durations(state_panel: pd.DataFrame, n_states: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = extract_runs(state_panel["state"])
    rows = []
    for state in range(n_states):
        durations = runs.loc[runs["state"] == state, "duration"]
        rows.append(
            {
                "state": f"state_{state}",
                "occurrences": len(durations),
                "mean_duration": durations.mean() if len(durations) else np.nan,
                "median_duration": durations.median() if len(durations) else np.nan,
                "min_duration": durations.min() if len(durations) else np.nan,
                "max_duration": durations.max() if len(durations) else np.nan,
                "std_duration": durations.std() if len(durations) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows), runs


def state_shares(state_panel: pd.DataFrame, n_states: int) -> pd.Series:
    shares = state_panel["state"].value_counts(normalize=True).sort_index()
    return shares.reindex(range(n_states), fill_value=0.0)


def plot_state_path(state_panel: pd.DataFrame, n_states: int, file_name: str) -> None:
    colors = sns.color_palette("tab10", n_states)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.step(state_panel["date"], state_panel["state"], where="post", color="#222222", linewidth=1.2)
    for state in range(n_states):
        mask = state_panel["state"] == state
        ax.fill_between(state_panel["date"], -0.5, n_states - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f"state_{i}" for i in range(n_states)])
    ax.set_title(f"{n_states}-State Jump Model Path")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=180)
    plt.close()


def plot_state_probabilities(state_panel: pd.DataFrame, n_states: int, file_name: str) -> None:
    colors = sns.color_palette("tab10", n_states)
    plt.figure(figsize=(12, 5.2))
    for state in range(n_states):
        plt.plot(state_panel["date"], state_panel[f"state_prob_{state}"], label=f"state_{state}", color=colors[state], linewidth=1.2)
    plt.title(f"{n_states}-State Jump Model Membership Strength")
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, ncol=min(n_states, 4))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=180)
    plt.close()


def plot_state_profile(state_panel: pd.DataFrame, n_states: int, file_name: str) -> None:
    candidate_vars = [
        "growth_pc1",
        "growth_pc2",
        "inflation_pc1",
        "inflation_pc2",
        "gs10",
        "term_spread_10y_1y",
        "bog_amom",
        "credit_spread",
        "ur_diff",
        "cp_amom",
        "hs_amom",
        "realized_vol",
    ]
    vars_present = [c for c in candidate_vars if c in state_panel.columns]
    profile = state_panel.groupby("state")[vars_present].mean()
    plt.figure(figsize=(10, 5.2 + 0.22 * len(vars_present)))
    sns.heatmap(profile.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title(f"{n_states}-State Mean Profile")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=180)
    plt.close()


def plot_durations(runs: pd.DataFrame, n_states: int, file_name: str) -> None:
    runs = runs.copy()
    runs["state_label"] = runs["state"].map(lambda x: f"state_{x}")
    plt.figure(figsize=(9, 4.8))
    sns.boxplot(data=runs, x="state_label", y="duration", color="#8ab6d6")
    sns.stripplot(data=runs, x="state_label", y="duration", color="#1f3b73", alpha=0.55, size=4)
    plt.title(f"{n_states}-State Duration Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=180)
    plt.close()


def run_one_state_count(panel: pd.DataFrame, X: np.ndarray, value_cols: list[str], n_states: int) -> dict[str, object]:
    model, search_df = select_best_model(X, n_states)
    best_row = search_df.loc[search_df["objective_value"].idxmin()]
    best_penalty = float(best_row["penalty"])
    best_seed = int(best_row["seed"])

    state_panel = build_state_panel(panel, model, n_states)
    state_summary = summarize_states(state_panel, value_cols)
    duration_summary, runs = summarize_durations(state_panel, n_states)

    state_panel.to_csv(RESULTS_DIR / f"jm_{n_states}_states_panel.csv", index=False)
    state_summary.to_csv(RESULTS_DIR / f"jm_state_summary_{n_states}.csv", index=False)
    duration_summary.to_csv(RESULTS_DIR / f"jm_duration_summary_{n_states}.csv", index=False)
    search_df.to_csv(RESULTS_DIR / f"jm_penalty_seed_search_{n_states}.csv", index=False)

    plot_state_path(state_panel, n_states, f"jm_{n_states}_state_path.png")
    plot_state_probabilities(state_panel, n_states, f"jm_{n_states}_state_probabilities.png")
    plot_state_profile(state_panel, n_states, f"jm_{n_states}_state_profile.png")
    plot_durations(runs, n_states, f"jm_{n_states}_durations.png")

    shares = state_shares(state_panel, n_states)
    return {
        "n_states": n_states,
        "best_seed": best_seed,
        "selected_penalty": best_penalty,
        "objective_value": float(best_row["objective_value"]),
        "completed": True,
        "average_state_duration": float(duration_summary["mean_duration"].mean()),
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
        "shortest_max_duration": float(duration_summary["max_duration"].min()),
    }


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    panel, X, feature_cols = load_and_preprocess_panel()
    comparison_rows = []
    for n_states in STATE_COUNTS:
        comparison_rows.append(run_one_state_count(panel, X, feature_cols, n_states))

    comparison = pd.DataFrame(comparison_rows).sort_values("n_states").reset_index(drop=True)
    comparison.to_csv(RESULTS_DIR / "jm_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
