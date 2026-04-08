from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"
EDA_DIR = ROOT / "results" / "appendix" / "eda" / "Final"


def ensure_dirs() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def load_panel(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / file_name, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def zscore_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        StandardScaler().fit_transform(df[features]),
        columns=features,
        index=df.index,
    )


def extract_two_pcs(df: pd.DataFrame, features: list[str], prefix: str) -> pd.DataFrame:
    model = df.dropna(subset=features).copy()
    X = zscore_frame(model, features)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    out = model[["date"]].copy()
    out[f"{prefix}_pc1"] = pcs[:, 0]
    out[f"{prefix}_pc2"] = pcs[:, 1]
    return out


def plot_final_panel(panel: pd.DataFrame) -> None:
    cols = [c for c in panel.columns if c != "date"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, max(10, 2.2 * len(cols))), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(panel["date"], panel[col], color="#1f3b73", linewidth=1.1)
        ax.axhline(0, color="0.75", linewidth=0.8, linestyle="--")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_series_panel.png", dpi=180)
    plt.close()


def plot_corr_heatmap(panel: pd.DataFrame) -> None:
    features = [c for c in panel.columns if c != "date"]
    corr = panel[features].corr()
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Final Macro Panel Correlation")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_corr_heatmap.png", dpi=180)
    plt.close()


def plot_missingness(original_frames: dict[str, pd.DataFrame], final_panel: pd.DataFrame) -> None:
    summary = []
    for name, df in original_frames.items():
        cols = [c for c in df.columns if c != "date"]
        valid = df.dropna(subset=cols)
        summary.append(
            {
                "panel": name,
                "start": valid["date"].min(),
                "end": valid["date"].max(),
                "rows": len(valid),
            }
        )
    summary.append(
        {
            "panel": "final_macro_inner_join",
            "start": final_panel["date"].min(),
            "end": final_panel["date"].max(),
            "rows": len(final_panel),
        }
    )
    summary_df = pd.DataFrame(summary)
    summary_df["start_num"] = summary_df["start"].map(pd.Timestamp.toordinal)
    summary_df["end_num"] = summary_df["end"].map(pd.Timestamp.toordinal)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(summary_df))
    ax.hlines(y, summary_df["start_num"], summary_df["end_num"], color="#3f8fc5", linewidth=6)
    ax.scatter(summary_df["start_num"], y, color="#1f3b73", s=30)
    ax.scatter(summary_df["end_num"], y, color="#b24a2f", s=30)
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df["panel"])
    ticks = np.linspace(summary_df["start_num"].min(), summary_df["end_num"].max(), 7)
    tick_labels = [pd.Timestamp.fromordinal(int(t)).strftime("%Y-%m") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=20)
    ax.set_title("Sample Coverage Before and After Inner Join")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_missingness.png", dpi=180)
    plt.close()

    text_df = summary_df[["panel", "start", "end", "rows"]].copy()
    text_df["start"] = text_df["start"].dt.strftime("%Y-%m-%d")
    text_df["end"] = text_df["end"].dt.strftime("%Y-%m-%d")
    text_df.to_csv(EDA_DIR / "final_macro_missingness_summary.csv", index=False)


def run_final_pca(panel: pd.DataFrame) -> tuple[float, float]:
    features = [c for c in panel.columns if c != "date"]
    X = zscore_frame(panel, features)
    pca = PCA()
    pcs = pca.fit_transform(X)

    explained = pd.DataFrame(
        {
            "pc": np.arange(1, len(features) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    loadings = pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"])
    scores = panel[["date"]].copy()
    scores["PC1"] = pcs[:, 0]
    scores["PC2"] = pcs[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(explained["pc"], explained["explained_variance_ratio"], marker="o", color="#1f3b73")
    plt.xticks(explained["pc"])
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Final Macro PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_pca_explained_variance.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        ax.bar(loadings.index, loadings[pc], color="#3f8fc5")
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_title(f"{pc} Loadings")
        ax.tick_params(axis="x", rotation=55)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_pca_loadings.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        ax.plot(scores["date"], scores[pc], color="#b24a2f", linewidth=1.2)
        ax.axhline(0, color="0.75", linestyle="--", linewidth=0.8)
        ax.set_title(f"{pc} Time Series")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "final_macro_pca_timeseries.png", dpi=180)
    plt.close()

    return float(explained.loc[0, "explained_variance_ratio"]), float(explained.loc[1, "explained_variance_ratio"])


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    growth = load_panel("growth_panel.csv")
    inflation = load_panel("inflation_panel.csv")
    rate = load_panel("rate_panel.csv")
    other = load_panel("other_panel.csv")

    growth_features = [c for c in growth.columns if c != "date"]
    inflation_features = [c for c in inflation.columns if c != "date"]

    growth_pcs = extract_two_pcs(growth, growth_features, "growth")
    inflation_pcs = extract_two_pcs(inflation, inflation_features, "inflation")

    final_panel = (
        growth_pcs
        .merge(inflation_pcs, on="date", how="inner")
        .merge(rate, on="date", how="inner")
        .merge(other, on="date", how="inner")
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    value_cols = [c for c in final_panel.columns if c != "date"]
    final_panel = final_panel.dropna(subset=value_cols).reset_index(drop=True)

    out = final_panel.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(PROCESSED_DIR / "final_macro_panel.csv", index=False)

    plot_final_panel(final_panel)
    plot_corr_heatmap(final_panel)
    plot_missingness(
        {
            "growth_pcs": growth_pcs,
            "inflation_pcs": inflation_pcs,
            "rate_panel": rate,
            "other_panel": other,
        },
        final_panel,
    )
    pc1, pc2 = run_final_pca(final_panel)

    print("Growth PCA features:", ", ".join(growth_features))
    print("Inflation PCA features:", ", ".join(inflation_features))
    print("Final panel columns:", ", ".join(final_panel.columns))
    print("Final inner-join range:", final_panel["date"].min().date(), final_panel["date"].max().date(), "rows", len(final_panel))
    print("Final PCA PC1:", round(pc1, 4))
    print("Final PCA PC2:", round(pc2, 4))


if __name__ == "__main__":
    main()
