from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
YFINANCE_VENDOR_DIR = ROOT / ".vendor_yf"
USER_SITE_GLOB = Path.home() / "AppData" / "Roaming" / "Python"
user_site_dirs: list[Path] = []
try:
    if USER_SITE_GLOB.exists():
        user_site_dirs.extend(USER_SITE_GLOB.glob("Python*/site-packages"))
    fallback_user_site = USER_SITE_GLOB / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages"
    if fallback_user_site.exists():
        user_site_dirs.append(fallback_user_site)
except PermissionError:
    user_site_dirs = []
for site_dir in user_site_dirs:
    if str(site_dir) not in sys.path:
        sys.path.append(str(site_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
if YFINANCE_VENDOR_DIR.exists() and str(YFINANCE_VENDOR_DIR) not in sys.path:
    # Append rather than prepend so core libs keep using the main environment.
    sys.path.append(str(YFINANCE_VENDOR_DIR))
import yfinance as yf


RESULTS_DIR = ROOT / "results" / "core" / "regime_interpretation"
PLOTS_DIR = RESULTS_DIR / "plots"
MAIN_PANEL_PATH = ROOT / "results" / "core" / "jump_model_penalty_grid" / "jump_model_penalty_grid_panel.csv"
FINAL_MACRO_PATH = ROOT / "data_processed" / "final_macro_panel.csv"
ASSETS_DIR = ROOT / "data_raw" / "assets"
TARGET_PENALTY = 0.6
MAIN_FEATURES = ["growth_pc1", "inflation_pc1", "gs10", "term_spread_10y_1y"]
EXTERNAL_VARS = ["credit_spread", "ur_diff", "bog_amom", "cp_amom", "hs_amom", "realized_vol"]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> None:
    for pattern in ["*.csv", "*.png"]:
        for path in RESULTS_DIR.glob(pattern):
            path.unlink()
    for path in PLOTS_DIR.glob("*.png"):
        path.unlink()


def load_main_regime_panel() -> pd.DataFrame:
    df = pd.read_csv(MAIN_PANEL_PATH, parse_dates=["date"])
    df = df.loc[np.isclose(df["penalty"], TARGET_PENALTY)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_regime_labels(regime_panel: pd.DataFrame) -> pd.DataFrame:
    means = regime_panel.groupby("state")[MAIN_FEATURES].mean()
    labels = pd.DataFrame(
        [
            {
                "state": 0,
                "regime_name": "Moderate Growth / Flat Curve Regime",
                "short_description": "Moderate growth, near-neutral inflation, mid-rate environment with a flatter curve.",
            },
            {
                "state": 1,
                "regime_name": "Low Inflation / Steep Curve Regime",
                "short_description": "Weaker growth, low inflation, moderate rates and a distinctly steeper curve.",
            },
            {
                "state": 2,
                "regime_name": "High Inflation / High Rate Regime",
                "short_description": "Elevated inflation with high long rates and a flatter curve backdrop.",
            },
        ]
    )
    labels["state"] = labels["state"].map(lambda x: f"state_{x}")
    labels["narrative_basis"] = labels["state"].map(
        {
            "state_0": f"growth_pc1={means.loc[0, 'growth_pc1']:.2f}, inflation_pc1={means.loc[0, 'inflation_pc1']:.2f}, gs10={means.loc[0, 'gs10']:.2f}, spread={means.loc[0, 'term_spread_10y_1y']:.2f}",
            "state_1": f"growth_pc1={means.loc[1, 'growth_pc1']:.2f}, inflation_pc1={means.loc[1, 'inflation_pc1']:.2f}, gs10={means.loc[1, 'gs10']:.2f}, spread={means.loc[1, 'term_spread_10y_1y']:.2f}",
            "state_2": f"growth_pc1={means.loc[2, 'growth_pc1']:.2f}, inflation_pc1={means.loc[2, 'inflation_pc1']:.2f}, gs10={means.loc[2, 'gs10']:.2f}, spread={means.loc[2, 'term_spread_10y_1y']:.2f}",
        }
    )
    return labels


def load_external_panel() -> pd.DataFrame:
    df = pd.read_csv(FINAL_MACRO_PATH, parse_dates=["date"])
    cols = ["date"] + EXTERNAL_VARS
    return df[cols].copy()


def build_external_validation(regime_panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_map = dict(zip(labels["state"], labels["regime_name"]))
    panel = regime_panel.copy()
    panel["state"] = panel["state"].map(lambda x: f"state_{int(x)}")
    panel["regime_name"] = panel["state"].map(label_map)
    panel = panel.merge(load_external_panel(), on="date", how="left")

    rows = []
    for regime_name, group in panel.groupby("regime_name"):
        for var in EXTERNAL_VARS:
            s = group[var].dropna()
            rows.append(
                {
                    "regime_name": regime_name,
                    "variable": var,
                    "mean": s.mean(),
                    "std": s.std(),
                    "count": s.count(),
                    "median": s.median(),
                    "p25": s.quantile(0.25),
                    "p75": s.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def plot_external_validation_heatmap(validation: pd.DataFrame) -> None:
    mat = validation.pivot(index="regime_name", columns="variable", values="mean").reindex(columns=EXTERNAL_VARS)
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("External Validation Means by Regime")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regime_external_validation_heatmap.png", dpi=180)
    plt.close()


def monthly_from_daily_price(path: Path, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[date_col])
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month", as_index=False)["price"].last().rename(columns={"month": "date"})
    return monthly


def monthly_from_series(path: Path, date_col: str, price_col: str, date_format: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    # Some asset files, especially gold, contain quoted prices with thousands
    # separators (for example "4,744.5"). Strip commas before numeric parsing
    # so the full monthly history is retained.
    price_str = df[price_col].astype(str).str.strip().str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(price_str, errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df[["date", "price"]]


def download_bond_monthly(start_date: pd.Timestamp) -> pd.DataFrame:
    bond = yf.download("VUSTX", start=start_date.strftime("%Y-%m-%d"), interval="1mo", auto_adjust=True, progress=False)
    if bond.empty:
        raise RuntimeError("Failed to download VUSTX monthly data.")
    if "Close" in bond.columns:
        series = bond["Close"]
    else:
        series = bond.iloc[:, 0]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    out = series.reset_index()
    out.columns = ["date", "price"]
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)


def load_assets(regime_panel: pd.DataFrame) -> pd.DataFrame:
    start = regime_panel["date"].min()
    spx = monthly_from_daily_price(ASSETS_DIR / "GSPC_yfinance_daily.csv", "date", "price").rename(columns={"price": "spx"})
    oil = monthly_from_series(ASSETS_DIR / "oil.csv", "observation_date", "WTISPLC").rename(columns={"price": "oil"})
    gold = monthly_from_series(ASSETS_DIR / "gold.csv", "date", "USD").rename(columns={"price": "gold"})
    bond = download_bond_monthly(start).rename(columns={"price": "bond"})

    panel = regime_panel[["date", "state"]].copy()
    for frame in [spx, oil, gold, bond]:
        panel = panel.merge(frame, on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


def compute_asset_performance(asset_panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_map = dict(zip(labels["state"], labels["regime_name"]))
    panel = asset_panel.copy()
    panel["state"] = panel["state"].map(lambda x: f"state_{int(x)}")
    panel["regime_name"] = panel["state"].map(label_map)

    rows = []
    for asset in ["spx", "oil", "bond", "gold"]:
        tmp = panel[["date", "regime_name", asset]].dropna().sort_values("date").copy()
        tmp["ret"] = tmp[asset].pct_change()
        tmp = tmp.dropna(subset=["ret"])
        for regime_name, group in tmp.groupby("regime_name"):
            returns = group["ret"]
            n = len(returns)
            wealth = (1 + returns).cumprod()
            twr = wealth.iloc[-1] ** (12 / n) - 1 if n > 0 and wealth.iloc[-1] > 0 else np.nan
            roll_max = wealth.cummax()
            drawdown = wealth / roll_max - 1
            ann_vol = returns.std() * np.sqrt(12) if n > 1 else np.nan
            rows.append(
                {
                    "asset": asset,
                    "regime_name": regime_name,
                    "n_months": n,
                    "time_weighted_annualized_return": twr,
                    "annualized_vol": ann_vol,
                    "sharpe_like": twr / ann_vol if pd.notna(ann_vol) and ann_vol not in [0, np.nan] else np.nan,
                    "hit_ratio": (returns > 0).mean(),
                    "max_drawdown": drawdown.min(),
                }
            )
    return pd.DataFrame(rows)


def regime_color_map() -> dict[str, str]:
    return {
        "Moderate Growth / Flat Curve Regime": "#dceaf6",
        "Low Inflation / Steep Curve Regime": "#f8f0db",
        "High Inflation / High Rate Regime": "#f3d1d1",
    }


def contiguous_regime_spans(asset_panel: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, object]]:
    label_map = dict(zip(labels["state"], labels["regime_name"]))
    panel = asset_panel[["date", "state"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    spans: list[dict[str, object]] = []
    if panel.empty:
        return spans
    start_idx = 0
    for i in range(1, len(panel)):
        if panel.loc[i, "state"] != panel.loc[i - 1, "state"]:
            spans.append(
                {
                    "state": int(panel.loc[start_idx, "state"]),
                    "regime_name": label_map.get(f"state_{int(panel.loc[start_idx, 'state'])}", f"state_{int(panel.loc[start_idx, 'state'])}"),
                    "start": panel.loc[start_idx, "date"],
                    "end": panel.loc[i, "date"],
                }
            )
            start_idx = i
    spans.append(
        {
            "state": int(panel.loc[start_idx, "state"]),
            "regime_name": label_map.get(f"state_{int(panel.loc[start_idx, 'state'])}", f"state_{int(panel.loc[start_idx, 'state'])}"),
            "start": panel.loc[start_idx, "date"],
            "end": panel.loc[len(panel) - 1, "date"] + pd.offsets.MonthBegin(1),
        }
    )
    return spans


def segment_stats(series: pd.DataFrame) -> tuple[float, float] | tuple[None, None]:
    tmp = series[["date", "price"]].dropna().sort_values("date").copy()
    if len(tmp) < 2:
        return None, None
    tmp["ret"] = tmp["price"].pct_change()
    tmp = tmp.dropna(subset=["ret"])
    if tmp.empty:
        return None, None
    wealth = (1 + tmp["ret"]).cumprod()
    ann_ret = wealth.iloc[-1] ** (12 / len(tmp)) - 1 if wealth.iloc[-1] > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    return ann_ret, drawdown.min()


def add_regime_background(ax: plt.Axes, spans: list[dict[str, object]], color_map: dict[str, str]) -> None:
    for span in spans:
        ax.axvspan(span["start"], span["end"], color=color_map.get(span["regime_name"], "#dddddd"), alpha=0.45, zorder=0)


def add_segment_annotations(ax: plt.Axes, asset_series: pd.DataFrame, spans: list[dict[str, object]]) -> None:
    for span in spans:
        segment = asset_series[(asset_series["date"] >= span["start"]) & (asset_series["date"] < span["end"])].copy()
        if len(segment) < 6:
            continue
        ann_ret, mdd = segment_stats(segment)
        if ann_ret is None or mdd is None or pd.isna(ann_ret) or pd.isna(mdd):
            continue
        mid = segment["date"].iloc[len(segment) // 2]
        y = np.log(segment["price"]).median()
        ax.text(
            mid,
            y,
            f"AnnRet: {ann_ret * 100:.1f}%\nMDD: {mdd * 100:.1f}%",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            zorder=3,
        )


def add_regime_legend(ax: plt.Axes, color_map: dict[str, str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.45) for color in color_map.values()]
    ax.legend(handles, list(color_map.keys()), title="Macro Regime", loc="upper left", frameon=True, framealpha=0.9)


def plot_asset_panels_with_regime_background(asset_panel: pd.DataFrame, labels: pd.DataFrame, filename: str) -> None:
    panel = asset_panel.copy()
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    title_map = {"spx": "S&P 500 Price with Regime Background", "oil": "Oil Price with Regime Background", "bond": "US Treasury Bonds Price with Regime Background", "gold": "Gold Price with Regime Background"}
    color_map = regime_color_map()
    spans = contiguous_regime_spans(panel, labels)
    global_end = panel["date"].max()

    for ax, asset in zip(axes, ["spx", "oil", "bond", "gold"]):
        series = panel[["date", asset]].dropna().rename(columns={asset: "price"}).sort_values("date")
        add_regime_background(ax, spans, color_map)
        ax.plot(series["date"], np.log(series["price"]), color="#1f77b4", linewidth=1.8, zorder=2)
        add_segment_annotations(ax, series, spans)
        ax.set_title(title_map[asset], fontsize=16)
        ax.set_ylabel("Log Price")
        ax.grid(True, alpha=0.35)
        add_regime_legend(ax, color_map)
        if not series.empty and series["date"].max() < global_end:
            end_date = series["date"].max()
            ax.axvline(end_date, color="#b24a2f", linestyle="--", linewidth=1.0)
            ax.text(end_date, ax.get_ylim()[1], f"data ends {end_date:%Y-%m}", ha="right", va="bottom", fontsize=8, color="#b24a2f")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    cleanup_outputs()
    sns.set_theme(style="whitegrid")

    regime_panel = load_main_regime_panel()
    labels = build_regime_labels(regime_panel)
    labels.to_csv(RESULTS_DIR / "regime_labels.csv", index=False)

    external = build_external_validation(regime_panel, labels)
    external.to_csv(RESULTS_DIR / "regime_external_validation_table.csv", index=False)
    plot_external_validation_heatmap(external)

    assets = load_assets(regime_panel)
    perf = compute_asset_performance(assets, labels)
    perf.to_csv(RESULTS_DIR / "asset_performance_by_regime.csv", index=False)
    plot_asset_panels_with_regime_background(assets, labels, "asset_price_overview.png")
    plot_asset_panels_with_regime_background(assets, labels, "asset_history_macro_events.png")

    print(labels.to_string(index=False))
    print(external.sort_values(["variable", "regime_name"]).to_string(index=False))
    print(perf.sort_values(["asset", "regime_name"]).to_string(index=False))


if __name__ == "__main__":
    main()
