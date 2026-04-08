from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

VENDOR = Path(__file__).resolve().parents[3] / ".vendor"
if str(VENDOR) not in sys.path:
    # Append vendor after the base scientific stack is imported so
    # matplotlib/pandas keep using the main environment.
    sys.path.append(str(VENDOR))

from jumpmodels.jump import JumpModel


ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
ASSETS_DIR = ROOT / "data_raw" / "assets"
RESULTS_DIR = ROOT / "results" / "extensions" / "stress_aware_credit_4state"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
    "credit_spread",
]
ALL_PROFILE_VARS = FEATURES + ["realized_vol", "spx_ret", "bond_ret"]
N_STATES = 4
SEEDS = list(range(10))
PENALTY = 1.0


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> None:
    for pattern in ["*.csv", "*.png"]:
        for path in RESULTS_DIR.glob(pattern):
            path.unlink()


def load_macro_panel() -> pd.DataFrame:
    return (
        pd.read_csv(INPUT_PATH, parse_dates=["date"])
        .sort_values("date")
        .loc[:, ["date", "realized_vol"] + FEATURES]
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )


def monthly_from_daily_price(path: Path, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("month", as_index=False)["price"].last().rename(columns={"month": "date"})


def monthly_from_series(path: Path, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    price_str = df[price_col].astype(str).str.strip().str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(price_str, errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df[["date", "price"]]


def load_asset_returns() -> pd.DataFrame:
    spx = monthly_from_daily_price(ASSETS_DIR / "GSPC_yfinance_daily.csv", "date", "price").rename(columns={"price": "spx"})
    bond = monthly_from_series(ASSETS_DIR / "VUSTX_monthly.csv", "date", "price").rename(columns={"price": "bond"})

    for frame, col in [(spx, "spx"), (bond, "bond")]:
        frame[f"{col}_ret"] = frame[col].pct_change()

    out = spx[["date", "spx_ret"]].merge(bond[["date", "bond_ret"]], on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def build_model_panel() -> pd.DataFrame:
    panel = load_macro_panel().merge(load_asset_returns(), on="date", how="left")
    return panel.sort_values("date").reset_index(drop=True)


def zscore_features(df: pd.DataFrame) -> np.ndarray:
    # The Jump Model is fit only on the standardized feature matrix.
    # Original-scale columns are retained for state interpretation.
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


def extract_runs(states: pd.Series) -> pd.DataFrame:
    run_id = (states != states.shift()).cumsum()
    return (
        pd.DataFrame({"state": states, "run_id": run_id})
        .groupby("run_id")
        .agg(state=("state", "first"), duration=("state", "size"))
        .reset_index(drop=True)
    )


def build_state_panel(panel: pd.DataFrame, model: JumpModel, penalty: float, best_seed: int) -> pd.DataFrame:
    out = panel.copy()
    out["penalty"] = penalty
    out["best_seed"] = best_seed
    out["state"] = np.asarray(model.labels_).astype(int)
    proba = np.asarray(model.proba_)
    for i in range(N_STATES):
        out[f"state_prob_{i}"] = proba[:, i]
    return out


def state_moments_long(state_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_count = len(state_panel)
    penalty = float(state_panel["penalty"].iloc[0])
    best_seed = int(state_panel["best_seed"].iloc[0])
    for state, group in state_panel.groupby("state"):
        share = len(group) / total_count
        for var in ALL_PROFILE_VARS:
            s = group[var].dropna()
            rows.append(
                {
                    "penalty": penalty,
                    "best_seed": best_seed,
                    "state": f"state_{int(state)}",
                    "variable": var,
                    "mean": float(s.mean()) if len(s) else np.nan,
                    "variance": float(s.var()) if len(s) else np.nan,
                    "std": float(s.std()) if len(s) else np.nan,
                    "count": int(s.count()),
                    "state_share": float(share),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    panel = build_model_panel()
    X = zscore_features(panel)
    model, best_seed, _objective_value = select_best_model(X, PENALTY)
    state_panel = build_state_panel(panel, model, PENALTY, best_seed)
    moments_long = state_moments_long(state_panel)
    moments_long.to_csv(RESULTS_DIR / "stress_aware_4state_penalty_1.5_state_moments.csv", index=False)
    print(moments_long.to_string(index=False))


if __name__ == "__main__":
    main()
