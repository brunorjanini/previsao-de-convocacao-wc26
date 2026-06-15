# -*- coding: utf-8 -*-
"""
Script 5 — Comparação consolidada de todos os modelos treinados.

Lê data/processed/model_comparison.csv e gera três gráficos em
data/processed/charts/comparison_all_models.png:

  Painel 1 — Barras agrupadas por branch (Pesquisa / Feira): todas as métricas
  Painel 2 — Radar / linha: cada algoritmo sobreposto por métrica
  Painel 3 — Ranking F1 (barra horizontal), destacando o melhor por branch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

COMPARISON_CSV = Path("../../data/processed/model_comparison.csv")
CHARTS_DIR     = Path("../../data/processed/charts")
OUTPUT         = CHARTS_DIR / "comparison_all_models.png"

ALGO_LABELS = {"dt": "Árvore de Decisão", "xgb": "XGBoost", "knn": "KNN"}
ALGO_COLORS = {"dt": "#4C72B0", "xgb": "#DD8452", "knn": "#55A868"}
BRANCH_LABELS = {"pesquisa": "Pesquisa (todas as features)", "feira": "Feira (6 notas + posição)"}
METRICS      = ["precision", "recall", "f1", "f1_macro"]
METRIC_LABELS = ["Precision", "Recall", "F1", "F1-Macro"]


def load():
    df = pd.read_csv(COMPARISON_CSV)
    df["model_label"] = df["model"].map(ALGO_LABELS)
    return df


def plot_grouped_bars(ax, df, branch):
    sub = df[df["branch"] == branch].set_index("model")
    models = list(ALGO_LABELS.keys())
    x = np.arange(len(METRICS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, model in enumerate(models):
        if model not in sub.index:
            continue
        vals = [sub.loc[model, m] for m in METRICS]
        bars = ax.bar(x + offsets[i], vals, width=width,
                      label=ALGO_LABELS[model], color=ALGO_COLORS[model], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title(BRANCH_LABELS[branch], fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=9)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_f1_ranking(ax, df):
    df_sorted = df.sort_values(["branch", "f1"], ascending=[True, False])

    labels, values, colors, hatches = [], [], [], []
    for _, row in df_sorted.iterrows():
        branch_short = "P" if row["branch"] == "pesquisa" else "F"
        labels.append(f"{ALGO_LABELS[row['model']]}  [{branch_short}]")
        values.append(row["f1"])
        colors.append(ALGO_COLORS[row["model"]])
        hatches.append("" if row["branch"] == "pesquisa" else "//")

    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, alpha=0.85,
                   hatch=hatches, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1-Score (classe positiva)", fontsize=9)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_title("Ranking F1 por Modelo e Branch", fontsize=12, fontweight="bold", pad=8)
    ax.axvline(0.3, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pesquisa_patch = mpatches.Patch(facecolor="white", edgecolor="gray", label="[P] Pesquisa")
    feira_patch    = mpatches.Patch(facecolor="white", edgecolor="gray", hatch="//", label="[F] Feira")
    ax.legend(handles=[pesquisa_patch, feira_patch], fontsize=8, loc="lower right")


def plot_metric_lines(ax, df):
    for branch, ls in [("pesquisa", "-"), ("feira", "--")]:
        sub = df[df["branch"] == branch].set_index("model")
        for model in ALGO_LABELS:
            if model not in sub.index:
                continue
            vals = [sub.loc[model, m] for m in METRICS]
            ax.plot(METRIC_LABELS, vals, marker="o", linestyle=ls,
                    color=ALGO_COLORS[model], linewidth=1.8, markersize=6,
                    label=f"{ALGO_LABELS[model]} ({'Pesquisa' if branch == 'pesquisa' else 'Feira'})")
            for j, val in enumerate(vals):
                ax.annotate(f"{val:.2f}", (j, val),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=ALGO_COLORS[model])

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Perfil de Métricas — Todos os Modelos", fontsize=12, fontweight="bold", pad=8)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Comparação Consolidada de Modelos — WC2026 Convocation Predictor",
                 fontsize=15, fontweight="bold", y=0.98)

    # Layout: linha 1 → dois painéis de barras agrupadas
    #         linha 2 → linha de perfil + ranking horizontal
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax_p  = fig.add_subplot(gs[0, 0])
    ax_f  = fig.add_subplot(gs[0, 1])
    ax_ln = fig.add_subplot(gs[1, 0])
    ax_rk = fig.add_subplot(gs[1, 1])

    plot_grouped_bars(ax_p, df, "pesquisa")
    plot_grouped_bars(ax_f, df, "feira")
    plot_metric_lines(ax_ln, df)
    plot_f1_ranking(ax_rk, df)

    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {OUTPUT}")

    print("\nResumo da comparação:")
    print(df[["branch", "model", "precision", "recall", "f1", "f1_macro"]].to_string(index=False))

    for branch in ["pesquisa", "feira"]:
        sub = df[df["branch"] == branch]
        best = sub.loc[sub["f1"].idxmax()]
        print(f"\nMelhor {branch}: {ALGO_LABELS[best['model']]}  (F1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
