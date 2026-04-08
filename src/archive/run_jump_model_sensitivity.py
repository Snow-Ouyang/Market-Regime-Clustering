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
RESULTS_DIR = ROOT / "results" / "appendix" / "archive_jump_model_sensitivity"
PLOTS_DIR = RESULTS_DIR / "plots"
N_STATES = 2
SEEDS = list(range(10))
PENALTIES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def penalty_to_str(penalty: float) -> str:
    return f"{penalty:g}"


def load_and_preprocess_panel() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    panel = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in panel.columns if c != "date"]
    panel = panel.dropna(subset=feature_cols).reset_index(drop=True)

    # Jump Model is fit on the standardized feature matrix only.
    # The original panel is preserved separately for interpretation, summaries, and exports.
    X = StandardScaler().fit_transform(panel[feature_cols])
    return panel, X, feature_cols


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


def select_best_model_for_penalty(X: np.ndarray, penalty: float) -> tuple[JumpModel, pd.DataFrame]:
    rows = []
    best_model = None
    best_obj = np.inf
    best_seed = None

    for seed in SEEDS:
        try:
            model = fit_one_jump_model(X, penalty, seed)
            objective = float(model.val_)
            rows.append(
                {
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
        except Exception as exc:
            rows.append(
                {
                    "penalty": penalty,
                    "seed": seed,
                    "objective_value": np.nan,
                    "completed": False,
                    "error": str(exc),
                }
            )

    if best_model is None:
        raise RuntimeError(f"No successful Jump Model fit for penalty={penalty}")

    search_df = pd.DataFrame(rows)
    search_df["best_seed"] = best_seed
    return best_model, search_df


def build_state_panel(panel: pd.DataFrame, model: JumpModel) -> pd.DataFrame:
    out = panel.copy()
    out["state"] = np.asarray(model.labels_).astype(int)

    proba = np.asarray(model.proba_)
    for i in range(N_STATES):
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


def summarize_durations(state_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = extract_runs(state_panel["state"])
    rows = []
    for state in range(N_STATES):
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


def state_shares(state_panel: pd.DataFrame) -> pd.Series:
    shares = state_panel["state"].value_counts(normalize=True).sort_index()
    return shares.reindex(range(N_STATES), fill_value=0.0)


def plot_state_path(state_panel: pd.DataFrame, penalty: float) -> None:
    label = penalty_to_str(penalty)
    colors = sns.color_palette("tab10", N_STATES)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.step(state_panel["date"], state_panel["state"], where="post", color="#222222", linewidth=1.2)
    for state in range(N_STATES):
        mask = state_panel["state"] == state
        ax.fill_between(state_panel["date"], -0.5, N_STATES - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
    ax.set_yticks(range(N_STATES))
    ax.set_yticklabels([f"state_{i}" for i in range(N_STATES)])
    ax.set_title(f"{N_STATES}-State Jump Model Path, penalty={label}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_path.png", dpi=180)
    plt.close()


def plot_state_probabilities(state_panel: pd.DataFrame, penalty: float) -> None:
    label = penalty_to_str(penalty)
    colors = sns.color_palette("tab10", N_STATES)
    plt.figure(figsize=(12, 5.2))
    for state in range(N_STATES):
        plt.plot(state_panel["date"], state_panel[f"state_prob_{state}"], label=f"state_{state}", color=colors[state], linewidth=1.2)
    plt.title(f"{N_STATES}-State Jump Model Membership Strength, penalty={label}")
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_probabilities.png", dpi=180)
    plt.close()


def plot_state_profile(state_panel: pd.DataFrame, penalty: float) -> None:
    label = penalty_to_str(penalty)
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
    plt.title(f"{N_STATES}-State Mean Profile, penalty={label}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_profile.png", dpi=180)
    plt.close()


def plot_durations(runs: pd.DataFrame, penalty: float) -> None:
    label = penalty_to_str(penalty)
    runs = runs.copy()
    runs["state_label"] = runs["state"].map(lambda x: f"state_{x}")
    plt.figure(figsize=(9, 4.8))
    sns.boxplot(data=runs, x="state_label", y="duration", color="#8ab6d6")
    sns.stripplot(data=runs, x="state_label", y="duration", color="#1f3b73", alpha=0.55, size=4)
    plt.title(f"{N_STATES}-State Duration Distribution, penalty={label}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_durations.png", dpi=180)
    plt.close()


def plot_sensitivity_summary(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("penalty").reset_index(drop=True)

    plt.figure(figsize=(8, 4.8))
    plt.plot(ordered["penalty"], ordered["min_state_share"], marker="o", label="min_state_share", color="#b24a2f")
    plt.plot(ordered["penalty"], ordered["max_state_share"], marker="o", label="max_state_share", color="#1f3b73")
    plt.xscale("symlog", linthresh=0.1)
    plt.title("Penalty vs State Share")
    plt.xlabel("jump penalty")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_vs_state_share.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(ordered["penalty"], ordered["avg_state_duration"], marker="o", color="#1f3b73")
    plt.xscale("symlog", linthresh=0.1)
    plt.title("Penalty vs Average State Duration")
    plt.xlabel("jump penalty")
    plt.ylabel("avg_state_duration")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_vs_avg_duration.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(ordered["penalty"], ordered["objective_value"], marker="o", color="#9c6b30")
    plt.xscale("symlog", linthresh=0.1)
    plt.title("Penalty vs Objective Value")
    plt.xlabel("jump penalty")
    plt.ylabel("objective_value")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"jm_{N_STATES}_states_penalty_vs_objective.png", dpi=180)
    plt.close()


def run_one_penalty(panel: pd.DataFrame, X: np.ndarray, value_cols: list[str], penalty: float) -> dict[str, object]:
    model, search_df = select_best_model_for_penalty(X, penalty)
    best_row = search_df.loc[search_df["objective_value"].idxmin()]
    best_seed = int(best_row["seed"])
    label = penalty_to_str(penalty)

    state_panel = build_state_panel(panel, model)
    state_summary = summarize_states(state_panel, value_cols)
    duration_summary, runs = summarize_durations(state_panel)

    state_panel.to_csv(RESULTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_panel.csv", index=False)
    state_summary.to_csv(RESULTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_state_summary.csv", index=False)
    duration_summary.to_csv(RESULTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_duration_summary.csv", index=False)
    search_df.to_csv(RESULTS_DIR / f"jm_{N_STATES}_states_penalty_{label}_seed_search.csv", index=False)

    plot_state_path(state_panel, penalty)
    plot_state_probabilities(state_panel, penalty)
    plot_state_profile(state_panel, penalty)
    plot_durations(runs, penalty)

    shares = state_shares(state_panel)
    return {
        "penalty": penalty,
        "best_seed": best_seed,
        "objective_value": float(best_row["objective_value"]),
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
        "smallest_state_mean_duration": float(duration_summary["mean_duration"].min()),
        "largest_state_mean_duration": float(duration_summary["mean_duration"].max()),
        "avg_state_duration": float(duration_summary["mean_duration"].mean()),
        "num_states_below_5pct_share": int((shares < 0.05).sum()),
        "num_states_below_3_month_mean_duration": int((duration_summary["mean_duration"] < 3).sum()),
    }


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    panel, X, value_cols = load_and_preprocess_panel()
    rows = []
    for penalty in PENALTIES:
        rows.append(run_one_penalty(panel, X, value_cols, penalty))

    summary = pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / f"jm_{N_STATES}_states_penalty_sensitivity_summary.csv", index=False)
    plot_sensitivity_summary(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
