from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

if str(VENDOR) not in sys.path:
    sys.path.append(str(VENDOR))

from jumpmodels.jump import JumpModel


FINAL_MACRO_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
ASSETS_DIR = ROOT / "data_raw" / "assets"
RESULTS_DIR = ROOT / "results" / "extensions" / "stress_aware_credit_4state"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
    "credit_spread",
]
ASSET_NAMES = ["spx", "oil", "bond", "gold"]
PENALTY = 1.5
N_STATES = 4
SEEDS = list(range(10))

REGIME_LABELS = {
    0: "Late-Cycle / Inflationary Flat Curve",
    1: "Low-Rate / Steep Curve",
    2: "High-Rate / Resilient Growth",
    3: "Macro-Financial Stress",
}

EVENT_SPANS = [
    ("First Oil Crisis / 1973-1975 recession", "1973-10-01", "1975-03-01", "#f4d6d6"),
    ("Volcker disinflation / 1980-1982 double-dip", "1980-01-01", "1982-11-01", "#d9e7f7"),
    ("1990-1991 recession / Gulf War / credit tightening", "1990-07-01", "1991-12-01", "#f8ead2"),
    ("Global Financial Crisis", "2007-12-01", "2009-06-01", "#ead7f7"),
    ("COVID shock", "2020-02-01", "2020-08-01", "#d8efe1"),
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> None:
    targets = [
        "stress_aware_4state_regime_panel.csv",
        "stress_aware_4state_asset_performance.csv",
        "stress_aware_4state_variable_moments.csv",
        "stress_aware_4state_regime_labels.csv",
    ]
    for name in targets:
        path = RESULTS_DIR / name
        if path.exists():
            path.unlink()
    for name in ["stress_aware_4state_state_path.png", "stress_aware_4state_asset_macro_events.png"]:
        path = PLOTS_DIR / name
        if path.exists():
            path.unlink()


def load_macro_panel() -> pd.DataFrame:
    df = pd.read_csv(FINAL_MACRO_PATH, parse_dates=["date"])
    keep_cols = ["date"] + FEATURES
    return df[keep_cols].dropna().sort_values("date").reset_index(drop=True)


def zscore_features(df: pd.DataFrame) -> np.ndarray:
    # The Jump Model is fit on standardized features only.
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
        raise RuntimeError("No successful stress-aware Jump Model fit.")
    return best_model, int(best_seed), float(best_obj)


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


def load_assets(regime_panel: pd.DataFrame) -> pd.DataFrame:
    spx = monthly_from_daily_price(ASSETS_DIR / "GSPC_yfinance_daily.csv", "date", "price").rename(columns={"price": "spx"})
    oil = monthly_from_series(ASSETS_DIR / "oil.csv", "observation_date", "WTISPLC").rename(columns={"price": "oil"})
    gold = monthly_from_series(ASSETS_DIR / "gold.csv", "date", "USD").rename(columns={"price": "gold"})
    bond = monthly_from_series(ASSETS_DIR / "VUSTX_monthly.csv", "date", "price").rename(columns={"price": "bond"})

    panel = regime_panel[["date", "state", "regime_name"]].copy()
    for frame in [spx, oil, bond, gold]:
        panel = panel.merge(frame, on="date", how="left")
    return panel.sort_values("date").reset_index(drop=True)


def compute_asset_performance(asset_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in ASSET_NAMES:
        tmp = asset_panel[["date", "regime_name", asset]].dropna().sort_values("date").copy()
        tmp["ret"] = tmp[asset].pct_change()
        tmp = tmp.dropna(subset=["ret"])
        for regime_name, group in tmp.groupby("regime_name"):
            returns = group["ret"]
            n = len(returns)
            wealth = (1 + returns).cumprod()
            twr = wealth.iloc[-1] ** (12 / n) - 1 if n > 0 and wealth.iloc[-1] > 0 else np.nan
            ann_vol = returns.std() * np.sqrt(12) if n > 1 else np.nan
            drawdown = wealth / wealth.cummax() - 1
            rows.append(
                {
                    "asset": asset,
                    "regime_name": regime_name,
                    "n_months": int(n),
                    "time_weighted_annualized_return": float(twr) if pd.notna(twr) else np.nan,
                    "annualized_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
                    "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_variable_moments(regime_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime_name, group in regime_panel.groupby("regime_name"):
        for var in FEATURES:
            s = group[var].dropna()
            rows.append(
                {
                    "regime_name": regime_name,
                    "variable": var,
                    "mean": float(s.mean()) if len(s) else np.nan,
                    "variance": float(s.var()) if len(s) else np.nan,
                    "std": float(s.std()) if len(s) else np.nan,
                    "count": int(s.count()),
                }
            )
    return pd.DataFrame(rows)


def plot_state_path(regime_panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    add_event_background(ax)
    ax.step(regime_panel["date"], regime_panel["state"], where="post", color="#333333", linewidth=1.1)
    ax.set_yticks(range(N_STATES))
    ax.set_yticklabels([REGIME_LABELS[i] for i in range(N_STATES)], fontsize=9)
    ax.set_title("Stress-Aware Extension: 4-State Jump Model Path")
    ax.grid(alpha=0.3)
    add_event_legend(ax)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stress_aware_4state_state_path.png", dpi=180)
    plt.close()


def regime_color_map() -> dict[str, str]:
    return {
        "Late-Cycle / Inflationary Flat Curve": "#dceaf6",
        "Low-Rate / Steep Curve": "#f8f0db",
        "High-Rate / Resilient Growth": "#f3d1d1",
        "Macro-Financial Stress": "#d8efe1",
    }


def contiguous_regime_spans(asset_panel: pd.DataFrame) -> list[dict[str, object]]:
    panel = asset_panel[["date", "regime_name"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    spans: list[dict[str, object]] = []
    if panel.empty:
        return spans
    start_idx = 0
    for i in range(1, len(panel)):
        if panel.loc[i, "regime_name"] != panel.loc[i - 1, "regime_name"]:
            spans.append(
                {
                    "regime_name": panel.loc[start_idx, "regime_name"],
                    "start": panel.loc[start_idx, "date"],
                    "end": panel.loc[i, "date"],
                }
            )
            start_idx = i
    spans.append(
        {
            "regime_name": panel.loc[start_idx, "regime_name"],
            "start": panel.loc[start_idx, "date"],
            "end": panel.loc[len(panel) - 1, "date"] + pd.offsets.MonthBegin(1),
        }
    )
    return spans


def add_regime_background(ax: plt.Axes, spans: list[dict[str, object]], color_map: dict[str, str]) -> None:
    for span in spans:
        ax.axvspan(span["start"], span["end"], color=color_map.get(span["regime_name"], "#dddddd"), alpha=0.4, zorder=0)


def add_regime_legend(ax: plt.Axes, color_map: dict[str, str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.4) for color in color_map.values()]
    ax.legend(handles, list(color_map.keys()), title="Macro Regimes", loc="upper left", frameon=True, framealpha=0.9)


def add_event_background(ax: plt.Axes) -> None:
    for label, start, end, color in EVENT_SPANS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.35, zorder=0)


def add_event_legend(ax: plt.Axes) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.35) for _, _, _, color in EVENT_SPANS]
    labels = [name for name, *_ in EVENT_SPANS]
    ax.legend(
        handles,
        labels,
        title="Macro Crisis Episodes",
        loc="upper right",
        frameon=True,
        framealpha=0.8,
        fontsize=6.5,
        title_fontsize=8,
        labelspacing=0.35,
        borderpad=0.35,
        handlelength=1.5,
    )


def plot_asset_panels_with_events(asset_panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    title_map = {
        "spx": "S&P 500 Price with Regime Background",
        "oil": "Oil Price with Regime Background",
        "bond": "US Treasury Bonds Price with Regime Background",
        "gold": "Gold Price with Regime Background",
    }
    spans = contiguous_regime_spans(asset_panel)
    color_map = regime_color_map()
    for ax, asset in zip(axes, ASSET_NAMES):
        series = asset_panel[["date", asset]].dropna().rename(columns={asset: "price"}).sort_values("date")
        add_regime_background(ax, spans, color_map)
        ax.plot(series["date"], np.log(series["price"]), color="#1f77b4", linewidth=1.8, zorder=2)
        ax.set_title(title_map[asset], fontsize=15)
        ax.set_ylabel("Log Price")
        ax.grid(True, alpha=0.35)
        add_regime_legend(ax, color_map)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stress_aware_4state_asset_macro_events.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    macro = load_macro_panel()
    X = zscore_features(macro)
    model, best_seed, objective_value = select_best_model(X)

    regime_panel = macro.copy()
    regime_panel["penalty"] = PENALTY
    regime_panel["best_seed"] = best_seed
    regime_panel["objective_value"] = objective_value
    regime_panel["state"] = np.asarray(model.labels_).astype(int)
    regime_panel["state_label"] = regime_panel["state"].map(lambda x: f"state_{x}")
    regime_panel["regime_name"] = regime_panel["state"].map(REGIME_LABELS)
    proba = np.asarray(model.proba_)
    for i in range(N_STATES):
        regime_panel[f"state_prob_{i}"] = proba[:, i]

    labels = pd.DataFrame(
        [{"state": f"state_{state}", "regime_name": name} for state, name in REGIME_LABELS.items()]
    )
    variable_moments = compute_variable_moments(regime_panel)
    assets = load_assets(regime_panel)
    asset_perf = compute_asset_performance(assets)

    regime_panel.to_csv(RESULTS_DIR / "stress_aware_4state_regime_panel.csv", index=False)
    labels.to_csv(RESULTS_DIR / "stress_aware_4state_regime_labels.csv", index=False)
    variable_moments.to_csv(RESULTS_DIR / "stress_aware_4state_variable_moments.csv", index=False)
    asset_perf.to_csv(RESULTS_DIR / "stress_aware_4state_asset_performance.csv", index=False)

    plot_state_path(regime_panel)
    plot_asset_panels_with_events(assets)

    print(labels.to_string(index=False))
    print(variable_moments.to_string(index=False))
    print(asset_perf.to_string(index=False))


if __name__ == "__main__":
    main()
