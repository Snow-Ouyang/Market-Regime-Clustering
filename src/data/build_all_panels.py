from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_raw" / "macro"
ASSETS = ROOT / "data_raw" / "assets"
PROCESSED = ROOT / "data_processed"
EDA = ROOT / "results" / "appendix" / "eda"


def ensure_dirs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in ["Growth", "Inflation", "Rate", "Risk", "Other"]:
        (EDA / name).mkdir(parents=True, exist_ok=True)


def standardize_series(path: Path, date_col: str, value_col: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    out["series"] = label
    return out


def infer_month_gap(series: pd.Series) -> int:
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 1
    months = int(round(diffs.dt.days.median() / 30.4375))
    return max(months, 1)


def annualized_percent_change(df: pd.DataFrame) -> pd.Series:
    gap = infer_month_gap(df["date"])
    growth = df["value"] / df["value"].shift(1)
    return (growth.pow(12 / gap) - 1) * 100


def zscore_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features, index=df.index)


def save_panel(panel: pd.DataFrame, name: str) -> None:
    out = panel.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(PROCESSED / name, index=False)


def plot_line(df: pd.DataFrame, col: str, title: str, path: Path, color: str = "#1f3b73", ylabel: str = "") -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df[col], color=color, linewidth=1.2)
    plt.axhline(0, color="0.75", linewidth=0.8, linestyle="--")
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_panel(df: pd.DataFrame, cols: list[str], titles: dict[str, str], path: Path) -> None:
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.3 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df["date"], df[col], color="#1f3b73", linewidth=1.2)
        ax.axhline(0, color="0.75", linewidth=0.8, linestyle="--")
        ax.set_title(titles[col])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, features: list[str], title: str, path: Path) -> None:
    model = df.dropna(subset=features).copy()
    corr = zscore_frame(model, features).corr()
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_pca(df: pd.DataFrame, features: list[str], prefix: str, out_dir: Path, clip_quantiles: tuple[float, float] | None = None) -> tuple[float, float]:
    model = df.dropna(subset=features).copy()
    if clip_quantiles is not None:
        lo_q, hi_q = clip_quantiles
        for feature in features:
            lo = model[feature].quantile(lo_q)
            hi = model[feature].quantile(hi_q)
            model[feature] = model[feature].clip(lo, hi)
    X = zscore_frame(model, features).to_numpy()
    pca = PCA()
    pcs = pca.fit_transform(X)

    explained = pd.DataFrame(
        {
            "pc": np.arange(1, len(features) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    loadings = pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"])
    scores = model[["date"]].copy()
    scores["PC1"] = pcs[:, 0]
    scores["PC2"] = pcs[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(explained["pc"], explained["explained_variance_ratio"], marker="o", color="#1f3b73")
    plt.xticks(explained["pc"])
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{prefix} PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower()}_pca_explained_variance.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        ax.bar(loadings.index, loadings[pc], color="#3f8fc5")
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_title(f"{pc} Loadings")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower()}_pca_loadings.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        ax.plot(scores["date"], scores[pc], color="#b24a2f", linewidth=1.2)
        ax.axhline(0, color="0.75", linestyle="--", linewidth=0.8)
        ax.set_title(f"{pc} Time Series")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower()}_pca_timeseries.png", dpi=180)
    plt.close()

    return float(explained.loc[0, "explained_variance_ratio"]), float(explained.loc[1, "explained_variance_ratio"])


def parse_ism() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Growth" / "ISM.csv").dropna(subset=["release_date_raw", "actual"]).copy()
    month_lookup = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
    pattern = r"(?P<release_month>[A-Za-z]{3}) \d{2}, (?P<release_year>\d{4}) \((?P<ref_month>[A-Za-z]{3})\)"

    def parse_reference_date(raw: str) -> pd.Timestamp:
        match = pd.Series([raw]).str.extract(pattern).iloc[0]
        if match.notna().all():
            release_month = month_lookup[match["release_month"]]
            release_year = int(match["release_year"])
            ref_month = month_lookup[match["ref_month"]]
            ref_year = release_year - 1 if ref_month > release_month else release_year
            return pd.Timestamp(year=ref_year, month=ref_month, day=1)
        fallback = pd.to_datetime(raw, errors="coerce")
        if pd.isna(fallback):
            raise ValueError(f"Unable to parse ISM date: {raw}")
        return (fallback - pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

    out = pd.DataFrame({"date": df["release_date_raw"].map(parse_reference_date), "value": pd.to_numeric(df["actual"], errors="coerce")})
    out = out.dropna().drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    out["series"] = "ism"
    return out


def build_growth() -> None:
    out_dir = EDA / "Growth"
    cfnai = standardize_series(RAW / "Growth" / "CFNAI revised.csv", "observation_date", "CFNAI", "cfnai")
    gdp = standardize_series(RAW / "Growth" / "GDP revised.csv", "observation_date", "GDPC1", "gdp")
    ipgr = standardize_series(RAW / "Growth" / "IPGR.csv", "period_start_date", "INDPRO", "ipgr")
    ism = parse_ism()

    cfnai = cfnai[["date", "value"]].rename(columns={"value": "cfnai"})
    gdp = gdp[["date", "value"]]
    gdp["gdp_amom"] = annualized_percent_change(gdp)
    gdp = gdp[["date", "gdp_amom"]]
    ipgr = ipgr[["date", "value"]]
    ipgr["ipgr_amom"] = annualized_percent_change(ipgr)
    ipgr = ipgr[["date", "ipgr_amom"]]
    ism = ism[["date", "value"]].rename(columns={"value": "ism"})

    monthly = pd.date_range(start=min(x["date"].min() for x in [cfnai, gdp, ipgr, ism]), end=max(x["date"].max() for x in [cfnai, gdp, ipgr, ism]), freq="MS")
    panel = pd.DataFrame({"date": monthly})
    for frame in [cfnai, gdp, ipgr, ism]:
        panel = panel.merge(frame, on="date", how="left")
    panel["gdp_amom"] = panel["gdp_amom"].ffill(limit=2)
    panel = panel[["date", "cfnai", "gdp_amom", "ipgr_amom", "ism"]]
    save_panel(panel, "growth_panel.csv")

    plot_panel(panel, ["cfnai", "gdp_amom", "ipgr_amom", "ism"], {
        "cfnai": "CFNAI Level",
        "gdp_amom": "GDP Annualized Growth",
        "ipgr_amom": "Industrial Production Annualized MoM Growth",
        "ism": "ISM Level",
    }, out_dir / "growth_series_panel.png")
    plot_corr_heatmap(panel, ["cfnai", "gdp_amom", "ipgr_amom", "ism"], "Growth Panel Correlation", out_dir / "growth_corr_heatmap.png")
    run_pca(panel, ["cfnai", "gdp_amom", "ipgr_amom", "ism"], "growth", out_dir)


def build_inflation() -> None:
    out_dir = EDA / "Inflation"
    cpi = standardize_series(RAW / "inflation" / "CPI.csv", "period_start_date", "CPIAUCSL", "cpi")
    ppi = standardize_series(RAW / "inflation" / "PPI revised.csv", "observation_date", "PPIACO", "ppi")
    si = standardize_series(RAW / "inflation" / "SI revised.csv", "observation_date", "UMCSENT", "si")

    cpi = cpi[["date", "value"]]
    cpi["cpi_amom"] = annualized_percent_change(cpi)
    cpi = cpi[["date", "cpi_amom"]]
    ppi = ppi[["date", "value"]]
    ppi["ppi_amom"] = annualized_percent_change(ppi)
    ppi = ppi[["date", "ppi_amom"]]
    si = si[["date", "value"]]
    si["si_diff"] = si["value"].diff()
    si = si[["date", "si_diff"]]

    monthly = pd.date_range(start=min(x["date"].min() for x in [cpi, ppi, si]), end=max(x["date"].max() for x in [cpi, ppi, si]), freq="MS")
    panel = pd.DataFrame({"date": monthly})
    for frame in [cpi, ppi, si]:
        panel = panel.merge(frame, on="date", how="left")
    panel = panel[["date", "cpi_amom", "ppi_amom", "si_diff"]]
    save_panel(panel, "inflation_panel.csv")

    plot_panel(panel, ["cpi_amom", "ppi_amom", "si_diff"], {
        "cpi_amom": "CPI Annualized MoM Inflation",
        "ppi_amom": "PPI Annualized MoM Inflation",
        "si_diff": "SI First Difference",
    }, out_dir / "inflation_series_panel.png")
    plot_corr_heatmap(panel, ["cpi_amom", "ppi_amom", "si_diff"], "Inflation Panel Correlation", out_dir / "inflation_corr_heatmap.png")
    run_pca(panel, ["cpi_amom", "ppi_amom", "si_diff"], "inflation", out_dir)


def build_rate() -> None:
    out_dir = EDA / "Rate"
    gs1 = standardize_series(RAW / "rate" / "GS1.csv", "observation_date", "GS1", "gs1")
    gs10 = standardize_series(RAW / "rate" / "GS10.csv", "observation_date", "GS10", "gs10")
    bog = standardize_series(RAW / "rate" / "BOG revised.csv", "observation_date", "BOGMBASE", "bog")

    gs1 = gs1[["date", "value"]].rename(columns={"value": "gs1"})
    gs10 = gs10[["date", "value"]].rename(columns={"value": "gs10"})
    bog = bog[["date", "value"]]
    bog["bog_amom"] = annualized_percent_change(bog)
    bog = bog[["date", "bog_amom"]]

    monthly = pd.date_range(start=min(x["date"].min() for x in [gs1, gs10, bog]), end=max(x["date"].max() for x in [gs1, gs10, bog]), freq="MS")
    panel = pd.DataFrame({"date": monthly})
    panel = panel.merge(gs1, on="date", how="left").merge(gs10, on="date", how="left").merge(bog, on="date", how="left")
    panel["term_spread_10y_1y"] = panel["gs10"] - panel["gs1"]
    panel = panel[["date", "gs10", "term_spread_10y_1y", "bog_amom"]]
    save_panel(panel, "rate_panel.csv")

    plt.figure(figsize=(11, 5.5))
    plt.plot(gs10["date"], gs10["gs10"], label="GS10", color="#1f3b73", linewidth=1.4)
    plt.plot(gs1["date"], gs1["gs1"], label="GS1", color="#9c6b30", linewidth=1.2)
    plt.title("GS10 and GS1 Levels")
    plt.ylabel("Percent")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "gs10_gs1_levels.png", dpi=180)
    plt.close()

    plot_line(panel, "term_spread_10y_1y", "Term Spread: GS10 - GS1", out_dir / "term_spread_10y_1y.png", "#b24a2f", "Percentage Points")
    plot_panel(panel, ["gs10", "term_spread_10y_1y", "bog_amom"], {
        "gs10": "GS10 Level",
        "term_spread_10y_1y": "Term Spread (10Y - 1Y)",
        "bog_amom": "BOG Annualized MoM Growth",
    }, out_dir / "rate_series_panel.png")
    plot_corr_heatmap(panel, ["gs10", "term_spread_10y_1y", "bog_amom"], "Rate Panel Correlation", out_dir / "rate_corr_heatmap.png")
    run_pca(panel, ["gs10", "term_spread_10y_1y", "bog_amom"], "rate", out_dir, clip_quantiles=(0.01, 0.99))


def build_risk() -> pd.DataFrame:
    out_dir = EDA / "Risk"
    prices = pd.read_csv(RAW / "Risk" / "GSPC_yfinance_daily.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["date", "price"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    prices["log_return"] = np.log(prices["price"]) - np.log(prices["price"].shift(1))
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp()
    prices["sq"] = np.square(prices["log_return"])
    panel = (
        prices.dropna(subset=["log_return"])
        .groupby("month", as_index=False)["sq"]
        .sum()
        .rename(columns={"month": "date", "sq": "realized_variance"})
    )
    panel["realized_vol"] = np.sqrt(panel["realized_variance"])

    plot_line(panel, "realized_vol", "Monthly Realized Volatility", out_dir / "realized_vol_timeseries.png", "#1f3b73", "Realized Volatility")
    plot_line(panel, "realized_variance", "Monthly Realized Variance", out_dir / "realized_variance_timeseries.png", "#9c6b30", "Realized Variance")
    log_panel = panel.copy()
    log_panel["log_realized_vol"] = np.log(log_panel["realized_vol"])
    plot_line(log_panel, "log_realized_vol", "Log Monthly Realized Volatility", out_dir / "log_realized_vol_timeseries.png", "#b24a2f", "log(realized_vol)")
    return panel[["date", "realized_variance", "realized_vol"]]


def build_other(risk_panel: pd.DataFrame) -> None:
    out_dir = EDA / "Other"
    aaa = standardize_series(RAW / "Credit" / "AAA.csv", "observation_date", "AAA", "aaa")
    baa = standardize_series(RAW / "Credit" / "BAA.csv", "observation_date", "BAA", "baa")
    ur = standardize_series(RAW / "labor" / "UR.csv", "period_start_date", "UNRATE", "ur")
    cp = standardize_series(RAW / "Other" / "CP.csv", "period_start_date", "CP", "cp")
    hs = standardize_series(RAW / "Other" / "HS.csv", "period_start_date", "HOUST", "hs")

    credit = aaa[["date", "value"]].rename(columns={"value": "aaa"}).merge(baa[["date", "value"]].rename(columns={"value": "baa"}), on="date", how="outer")
    credit["credit_spread"] = credit["baa"] - credit["aaa"]
    credit = credit[["date", "credit_spread"]]

    ur = ur[["date", "value"]]
    ur["ur_diff"] = ur["value"].diff()
    ur = ur[["date", "ur_diff"]]

    cp = cp[["date", "value"]]
    cp["cp_amom"] = annualized_percent_change(cp)
    cp = cp[["date", "cp_amom"]]

    hs = hs[["date", "value"]]
    hs["hs_amom"] = annualized_percent_change(hs)
    hs = hs[["date", "hs_amom"]]

    monthly = pd.date_range(start=min(x["date"].min() for x in [credit, ur, cp, hs]), end=max(x["date"].max() for x in [credit, ur, cp, hs]), freq="MS")
    panel = pd.DataFrame({"date": monthly})
    for frame in [credit, ur, cp, hs]:
        panel = panel.merge(frame, on="date", how="left")
    panel["cp_amom"] = panel["cp_amom"].ffill(limit=2)
    panel = panel.merge(risk_panel[["date", "realized_vol"]], on="date", how="left")
    panel = panel[["date", "credit_spread", "ur_diff", "cp_amom", "hs_amom", "realized_vol"]]
    save_panel(panel, "other_panel.csv")

    plot_line(panel, "credit_spread", "Credit Spread (BAA - AAA)", out_dir / "credit_spread_timeseries.png", "#7b2f2f", "Spread")
    plot_line(panel, "ur_diff", "UR First Difference", out_dir / "ur_diff_timeseries.png", "#1f3b73", "Difference")
    plot_line(panel, "cp_amom", "CP Annualized Native-Frequency Growth", out_dir / "cp_amom_timeseries.png", "#9c6b30", "Percent")
    plot_line(panel, "hs_amom", "Housing Starts Annualized MoM Growth", out_dir / "hs_amom_timeseries.png", "#3f8fc5", "Percent")
    plot_line(panel, "realized_vol", "Monthly Realized Volatility", out_dir / "realized_vol_timeseries.png", "#7b2f2f", "Realized Volatility")
    plot_panel(panel, ["credit_spread", "ur_diff", "cp_amom", "hs_amom", "realized_vol"], {
        "credit_spread": "Credit Spread (BAA - AAA)",
        "ur_diff": "UR First Difference",
        "cp_amom": "CP Annualized Native-Frequency Growth",
        "hs_amom": "Housing Starts Annualized MoM Growth",
        "realized_vol": "Monthly Realized Volatility",
    }, out_dir / "other_series_panel.png")
    plot_corr_heatmap(
        panel,
        ["credit_spread", "ur_diff", "cp_amom", "hs_amom", "realized_vol"],
        "Other Panel Correlation",
        out_dir / "other_corr_heatmap.png",
    )
    run_pca(
        panel,
        ["credit_spread", "ur_diff", "cp_amom", "hs_amom", "realized_vol"],
        "other",
        out_dir,
    )


def cleanup_unused_visuals() -> None:
    for path in [
        EDA / "Inflation" / "ppi_vintage_vs_revised.png",
        EDA / "Inflation" / "si_vintage_vs_revised.png",
        EDA / "Rate" / "bog_vintage_vs_revised.png",
        EDA / "Other" / "realized_variance_timeseries.png",
    ]:
        if path.exists():
            path.unlink()


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    build_growth()
    build_inflation()
    build_rate()
    risk_panel = build_risk()
    build_other(risk_panel)
    cleanup_unused_visuals()
    print("Unified panel build completed.")


if __name__ == "__main__":
    main()
