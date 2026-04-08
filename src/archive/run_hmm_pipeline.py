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
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
RESULTS_DIR = ROOT / "results" / "appendix" / "archive_hmm_full"
PLOTS_DIR = RESULTS_DIR / "plots"
SEEDS = list(range(15))
STATE_COUNTS = [2, 3, 4]
HMM_FEATURES = [
    "growth_pc1",
    "growth_pc2",
    "inflation_pc1",
    "inflation_pc2",
    "gs10",
    "term_spread_10y_1y",
    "credit_spread",
    "ur_diff",
    "realized_vol",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_panel() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    panel = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_cols = HMM_FEATURES.copy()
    panel = panel.dropna(subset=feature_cols).reset_index(drop=True)

    # HMM is fit on the standardized feature matrix only.
    # The original panel is preserved separately for state interpretation and summaries.
    scaler = StandardScaler()
    X = scaler.fit_transform(panel[feature_cols])
    X_df = pd.DataFrame(X, columns=feature_cols, index=panel.index)
    return panel, X_df, X, feature_cols


def fit_one_hmm(X: np.ndarray, n_states: int, seed: int) -> GaussianHMM:
    # Use diagonal covariance as the more stable baseline:
    # with 12 features and only ~600 months, full covariance is materially easier to overfit.
    model = GaussianHMM(
        n_components=n_states,
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


def select_best_model(X: np.ndarray, n_states: int) -> tuple[GaussianHMM, int, pd.DataFrame]:
    results = []
    best_model = None
    best_seed = None
    best_score = -np.inf

    for seed in SEEDS:
        try:
            model = fit_one_hmm(X, n_states, seed)
            score = model.score(X)
            converged = bool(model.monitor_.converged)
            iterations = int(model.monitor_.iter)
            results.append({"seed": seed, "log_likelihood": score, "converged": converged, "iterations": iterations})
            if score > best_score:
                best_score = score
                best_model = model
                best_seed = seed
        except Exception as exc:
            results.append({"seed": seed, "log_likelihood": np.nan, "converged": False, "iterations": np.nan, "error": str(exc)})

    if best_model is None:
        raise RuntimeError(f"No successful HMM fit for n_states={n_states}")

    return best_model, int(best_seed), pd.DataFrame(results)


def build_state_panel(panel: pd.DataFrame, X: np.ndarray, model: GaussianHMM, n_states: int) -> pd.DataFrame:
    states = model.predict(X)
    probs = model.predict_proba(X)
    out = panel.copy()
    out["state"] = states
    for i in range(n_states):
        out[f"state_prob_{i}"] = probs[:, i]
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


def get_aic_bic(model: GaussianHMM, X: np.ndarray) -> tuple[float, float]:
    if hasattr(model, "aic") and hasattr(model, "bic"):
        return float(model.aic(X)), float(model.bic(X))
    raise AttributeError("GaussianHMM aic/bic methods are unavailable in this hmmlearn build.")


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
    ax.set_title(f"{n_states}-State HMM Most Likely State Path")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=180)
    plt.close()


def plot_state_probabilities(state_panel: pd.DataFrame, n_states: int, file_name: str) -> None:
    colors = sns.color_palette("tab10", n_states)
    plt.figure(figsize=(12, 5.2))
    for state in range(n_states):
        plt.plot(state_panel["date"], state_panel[f"state_prob_{state}"], label=f"state_{state}", color=colors[state], linewidth=1.2)
    plt.title(f"{n_states}-State HMM Posterior Probabilities")
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
        "credit_spread",
        "ur_diff",
        "realized_vol",
    ]
    vars_present = [c for c in candidate_vars if c in state_panel.columns]
    profile = state_panel.groupby("state")[vars_present].mean()
    plt.figure(figsize=(10, 4.8 + 0.25 * len(vars_present)))
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
    model, best_seed, seed_results = select_best_model(X, n_states)
    state_panel = build_state_panel(panel, X, model, n_states)
    state_summary = summarize_states(state_panel, value_cols)
    duration_summary, runs = summarize_durations(state_panel, n_states)

    state_panel.to_csv(RESULTS_DIR / f"hmm_{n_states}_states_panel.csv", index=False)
    state_summary.to_csv(RESULTS_DIR / f"hmm_state_summary_{n_states}.csv", index=False)
    duration_summary.to_csv(RESULTS_DIR / f"hmm_duration_summary_{n_states}.csv", index=False)
    seed_results.to_csv(RESULTS_DIR / f"hmm_seed_search_{n_states}.csv", index=False)

    plot_state_path(state_panel, n_states, f"hmm_{n_states}_state_path.png")
    plot_state_probabilities(state_panel, n_states, f"hmm_{n_states}_state_probabilities.png")
    plot_state_profile(state_panel, n_states, f"hmm_{n_states}_state_profile.png")
    plot_durations(runs, n_states, f"hmm_{n_states}_durations.png")

    aic, bic = get_aic_bic(model, X)
    shares = state_shares(state_panel, n_states)
    return {
        "n_states": n_states,
        "best_seed": best_seed,
        "log_likelihood": float(model.score(X)),
        "AIC": aic,
        "BIC": bic,
        "converged": bool(model.monitor_.converged),
        "iterations": int(model.monitor_.iter),
        "average_state_duration": float(duration_summary["mean_duration"].mean()),
        "min_state_share": float(shares.min()),
        "max_state_share": float(shares.max()),
        "shortest_max_duration": float(duration_summary["max_duration"].min()),
    }


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    panel, _, X, feature_cols = load_and_preprocess_panel()
    comparison_rows = []

    for n_states in STATE_COUNTS:
        comparison_rows.append(run_one_state_count(panel, X, feature_cols, n_states))

    comparison = pd.DataFrame(comparison_rows).sort_values("n_states").reset_index(drop=True)
    comparison.to_csv(RESULTS_DIR / "hmm_model_comparison.csv", index=False)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
