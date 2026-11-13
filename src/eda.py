import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prepare_data import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "raw" / "ai4i2020.csv")
    splits = json.loads(
        (REPORTS / "split_indices.json").read_text(
            encoding="utf-8"
        )
    )

    # Ayrintili EDA yalnizca egitim kayitlariyla yapilir.
    train = df.loc[splits["train"]].copy()
    numeric = FEATURES[1:]

    sns.set_theme(style="whitegrid")

    # 1. Egitim verisinde sinif dagilimi
    counts = train[TARGET].value_counts().reindex(
        [0, 1], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Normal", "Failure"],
        counts.values,
        color=["#2457d6", "#c73545"],
    )
    ax.bar_label(bars, padding=4)
    ax.set_ylabel("Record count")
    ax.set_title("Training Set Class Distribution")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Sayisal degisken dagilimlari
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.flat, numeric):
        sns.histplot(
            data=train,
            x=column,
            bins=30,
            color="#2457d6",
            ax=ax,
        )
        ax.set_title(column)

    axes.flat[-1].axis("off")
    fig.suptitle("Training Set Feature Distributions")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_distributions.png", dpi=180)
    plt.close(fig)

    # 3. Korelasyonlar nedensellik gostermez.
    correlation = train[numeric + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Training Set Pearson Correlations")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    # 4. Normal ve arizali kayitlarin karsilastirilmasi
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, column in zip(axes.flat, numeric):
        sns.boxplot(
            data=train,
            x=TARGET,
            y=column,
            color="#99b9ec",
            ax=ax,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Failure"])
        ax.set_xlabel("")
        ax.set_title(column)

    axes.flat[-1].axis("off")
    fig.suptitle("Training Set Features by Failure Class")
    fig.tight_layout()
    fig.savefig(FIGURES / "features_by_class.png", dpi=180)
    plt.close(fig)

    # IQR siniri disindaki degerler sadece isaretlenir;
    # ariza belirtisi olabileceklerinden otomatik silinmez.
    outliers = []
    for column in numeric:
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = (
            (train[column] < lower) |
            (train[column] > upper)
        ).sum()

        outliers.append({
            "feature": column,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "flagged_count": int(count),
        })

    pd.DataFrame(outliers).to_csv(
        REPORTS / "training_outlier_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train[FEATURES + [TARGET]].describe(
        include="all"
    ).to_csv(
        REPORTS / "training_statistics.csv",
        encoding="utf-8-sig",
    )

    print(f"EDA tamamlandi. Egitim kaydi: {len(train)}")
    print(f"Normal: {int(counts.loc[0])}")
    print(f"Arizali: {int(counts.loc[1])}")
    print("\nIQR ile isaretlenen degerler:")
    print(pd.DataFrame(outliers).to_string(index=False))
    print("\n4 grafik reports/figures klasorune kaydedildi.")


if __name__ == "__main__":
    main()