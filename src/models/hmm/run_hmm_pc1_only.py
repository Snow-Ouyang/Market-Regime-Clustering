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
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
RESULTS_DIR = ROOT / "results" / "appendix" / "hmm_pc1_only"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
]
SEEDS = list(range(15))
N_STATES = 3


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> None:
    for pattern in ["*.csv", "*.png"]:
        for path in RESULTS_DIR.glob(pattern):
            path.unlink()
    for path in PLOTS_DIR.glob("*.png"):
        path.unlink()


def load_panel() -> tuple[pd.DataFrame, np.ndarray]:
    panel = (
        pd.read_csv(INPUT_PATH, parse_dates=["date"])
        .sort_values("date")
        .loc[:, ["date"] + FEATURES]
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )

    # HMM is fit on the standardized feature matrix only.
    # The original-scale panel is kept for interpretation and summaries.
    X = StandardScaler().fit_transform(panel[FEATURES])
    return panel, X


def fit_one_hmm(X: np.ndarray, seed: int) -> GaussianHMM:
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=500,
        tol=1e-4,
        min_covar=1e-4,
        random_state=seed,
        init_params="stmc",
        params="stmc",
    )
    model.fit(X)
    return model


def select_best_model(X: np.ndarray) -> tuple[GaussianHMM, pd.DataFrame]:
    rows = []
    best_model = None
    best_score = -np.inf
    for seed in SEEDS:
        try:
            model = fit_one_hmm(X, seed)
            score = float(model.score(X))
            rows.append(
                {
                    "seed": seed,
                    "log_likelihood": score,
                    "converged": bool(model.monitor_.converged),
                    "iterations": int(model.monitor_.iter),
                }
            )
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as exc:
            rows.append(
                {
                    "seed": seed,
                    "log_likelihood": np.nan,
                    "converged": False,
                    "iterations": np.nan,
                    "error": str(exc),
                }
            )
    if best_model is None:
        raise RuntimeError("No successful HMM fit.")
    return best_model, pd.DataFrame(rows)


def build_state_panel(panel: pd.DataFrame, X: np.ndarray, model: GaussianHMM) -> pd.DataFrame:
    out = panel.copy()
    out["state"] = model.predict(X)
    probs = model.predict_proba(X)
    for i in range(N_STATES):
        out[f"state_prob_{i}"] = probs[:, i]
    return out


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    return (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )


def summarize_result(state_panel: pd.DataFrame, model: GaussianHMM, X: np.ndarray, best_seed: int) -> pd.DataFrame:
    shares = state_panel["state"].value_counts(normalize=True).sort_index().reindex(range(N_STATES), fill_value=0.0)
    runs = extract_runs(state_panel["state"])
    durations = {}
    occurrences = {}
    for state in range(N_STATES):
        d = runs.loc[runs["state"] == state, "duration"]
        durations[state] = float(d.mean()) if len(d) else np.nan
        occurrences[state] = int(len(d))
    return pd.DataFrame(
        [
            {
                "n_states": N_STATES,
                "best_seed": best_seed,
                "log_likelihood": float(model.score(X)),
                "AIC": float(model.aic(X)),
                "BIC": float(model.bic(X)),
                "converged": bool(model.monitor_.converged),
                "iterations": int(model.monitor_.iter),
                "state_0_share": float(shares.loc[0]),
                "state_1_share": float(shares.loc[1]),
                "state_2_share": float(shares.loc[2]),
                "min_state_share": float(shares.min()),
                "max_state_share": float(shares.max()),
                "state_0_mean_duration": durations[0],
                "state_1_mean_duration": durations[1],
                "state_2_mean_duration": durations[2],
                "state_0_occurrences": occurrences[0],
                "state_1_occurrences": occurrences[1],
                "state_2_occurrences": occurrences[2],
            }
        ]
    )


def plot_state_path(state_panel: pd.DataFrame) -> None:
    colors = sns.color_palette("tab10", N_STATES)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.step(state_panel["date"], state_panel["state"], where="post", color="#222222", linewidth=1.2)
    for state in range(N_STATES):
        mask = state_panel["state"] == state
        ax.fill_between(state_panel["date"], -0.5, N_STATES - 0.5, where=mask, color=colors[state], alpha=0.18, step="post")
    ax.set_yticks(range(N_STATES))
    ax.set_yticklabels([f"state_{i}" for i in range(N_STATES)])
    ax.set_title("3-State HMM Path: pc1-only macro panel")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hmm_pc1_only_path.png", dpi=180)
    plt.close()


def plot_probabilities(state_panel: pd.DataFrame) -> None:
    colors = sns.color_palette("tab10", N_STATES)
    plt.figure(figsize=(12, 5.2))
    for state in range(N_STATES):
        plt.plot(state_panel["date"], state_panel[f"state_prob_{state}"], label=f"state_{state}", color=colors[state], linewidth=1.2)
    plt.title("3-State HMM Posterior Probabilities: pc1-only macro panel")
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hmm_pc1_only_probabilities.png", dpi=180)
    plt.close()


def plot_profile(state_panel: pd.DataFrame) -> None:
    profile = state_panel.groupby("state")[FEATURES].mean()
    plt.figure(figsize=(8.4, 5.2))
    sns.heatmap(profile.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("3-State HMM Mean Profile: pc1-only macro panel")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hmm_pc1_only_profile.png", dpi=180)
    plt.close()


def plot_durations(state_panel: pd.DataFrame) -> None:
    runs = extract_runs(state_panel["state"]).copy()
    runs["state_label"] = runs["state"].map(lambda x: f"state_{x}")
    plt.figure(figsize=(9, 4.8))
    sns.boxplot(data=runs, x="state_label", y="duration", color="#8ab6d6")
    sns.stripplot(data=runs, x="state_label", y="duration", color="#1f3b73", alpha=0.55, size=4)
    plt.title("3-State HMM Duration Distribution: pc1-only macro panel")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hmm_pc1_only_durations.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    panel, X = load_panel()
    model, seed_results = select_best_model(X)
    best_seed = int(seed_results.loc[seed_results["log_likelihood"].idxmax(), "seed"])
    state_panel = build_state_panel(panel, X, model)
    summary = summarize_result(state_panel, model, X, best_seed)

    state_panel.to_csv(RESULTS_DIR / "hmm_pc1_only_panel.csv", index=False)
    summary.to_csv(RESULTS_DIR / "hmm_pc1_only_summary.csv", index=False)
    seed_results.to_csv(RESULTS_DIR / "hmm_pc1_only_seed_search.csv", index=False)

    plot_state_path(state_panel)
    plot_probabilities(state_panel)
    plot_profile(state_panel)
    plot_durations(state_panel)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
