"""Figura: onde o pipeline falha. Tres cortes da MESMA particao retida (286 filamentos
agrupados por consenso de anotadores, 35 imagens, modelo seg@2048 alvo union, conf 0.20)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.55))

def binom(k, n): p = k / n; return p, np.sqrt(p * (1 - p) / n)

# (a) recall vs suporte de anotadores
# imagens multi-anotador; a barra hachurada e a populacao de anotador unico, onde
# "suporte 1.00" significa apenas que o unico especialista disponivel marcou o filamento
sup = [("0.33", 13, 42, 0), ("0.50", 9, 38, 0), ("0.67", 8, 12, 0),
       ("1.00", 25, 29, 0), ("1.00", 111, 165, 1)]
x = np.arange(len(sup))
r = [binom(k, n) for _, k, n, _ in sup]
ax[0].bar(x, [p for p, _ in r], yerr=[e for _, e in r],
          color=["#4c72b0"] * 4 + ["#8fb3d9"], alpha=0.85, capsize=3, width=0.7,
          hatch=["", "", "", "", "//"], edgecolor="#1f4e79", linewidth=0.7,
          error_kw={"lw": 0.9})
for i, (_, k, n, _) in enumerate(sup):
    ax[0].text(i, r[i][0] + r[i][1] + 0.03, f"{n}", ha="center", fontsize=6.6, color="0.4")
ax[0].set_xticks(x); ax[0].set_xticklabels([l for l, _, _, _ in sup], fontsize=7.5)
ax[0].axvline(3.5, color="0.55", lw=0.8, ls=":")
ax[0].text(2.0, 0.955, "2-3 experts", ha="center", fontsize=6.6, color="0.35")
ax[0].text(4.0, 0.955, "1 expert", ha="center", fontsize=6.6, color="0.35")
ax[0].set_xlabel("Annotator support")
ax[0].set_ylabel("Recall  (IoU > 0.5)")
ax[0].set_ylim(0, 1.0)

# (b) recall vs area do filamento (quintis)
sz = [("<0.5k", 0.386, 57), ("0.5-0.8k", 0.526, 57), ("0.8-1.3k", 0.649, 57),
      ("1.3-2.6k", 0.719, 57), (">2.6k", 0.638, 58)]
x = np.arange(len(sz))
e = [np.sqrt(p * (1 - p) / n) for _, p, n in sz]
ax[1].bar(x, [p for _, p, _ in sz], yerr=e, color="#55a868", alpha=0.85, capsize=3,
          width=0.7, error_kw={"lw": 0.9})
ax[1].set_xticks(x); ax[1].set_xticklabels([s for s, _, _ in sz], fontsize=6.6, rotation=30)
ax[1].set_xlabel("Filament area (px)")
ax[1].set_ylim(0, 0.95)

# (c) melhor IoU dos GT nao casados, em faixas nomeadas
lab = ["no\noverlap", "0.01-0.2", "0.2-0.4", "0.4-0.5"]
cnt = [67, 5, 24, 23]
col = ["#c44e52", "#8c8c8c", "#8c8c8c", "#dd8452"]
x = np.arange(4)
ax[2].bar(x, cnt, width=0.68, color=col, alpha=0.9)
for i, c_ in enumerate(cnt):
    ax[2].text(i, c_ + 1.8, f"{c_}", ha="center", fontsize=7, color="0.35")
ax[2].set_xticks(x); ax[2].set_xticklabels(lab, fontsize=7)
ax[2].set_xlabel("Best IoU with any prediction")
ax[2].set_ylabel("Unmatched GT filaments", fontsize=8.5)
ax[2].set_ylim(0, 80)
ax[2].text(0.48, 52, "detection\nfailure", ha="left", fontsize=7, color="#c44e52")
ax[2].text(3.0, 30, "mask\nquality", ha="center", fontsize=7, color="#dd8452")

for a_, l in zip(ax, "abc"):
    a_.text(0.04, 0.96, f"({l})", transform=a_.transAxes, fontsize=9,
            fontweight="bold", va="top")
fig.tight_layout(pad=0.5, w_pad=1.4)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/failures.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
