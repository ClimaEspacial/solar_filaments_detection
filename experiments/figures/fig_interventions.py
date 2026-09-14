"""Figura: seis eixos de intervencao independentes, todos com resultado nulo.

Todos os valores sao PQ na MESMA particao retida (35 imagens, 339 registros de anotador),
com a mesma implementacao de metrica. Marcador cheio = configuracao tambem pontuada no
leaderboard publico; todas as sete pontuadas deram 0.36.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE, NOISE = 0.4436, 0.009

G = [
 ("Detection threshold",   [("conf 0.15", 0.4321, 1), ("conf 0.20", 0.4436, 1), ("conf 0.30", 0.4450, 1)]),
 ("Annotator target",      [("single", 0.4477, 1), ("majority", 0.4493, 1), ("union", 0.4436, 1)]),
 ("Detector architecture", [("box, 9M, 1024 px", 0.4477, 1), ("seg, 27M, 2048 px", 0.4493, 1)]),
 ("Refiner capacity",      [("U-Net, 8.8M", 0.4436, 1), ("ResNet18, 15.4M", 0.4425, 0)]),
 ("Learned re-scorer",     [("disabled", 0.4321, 0), ("cut 0.20", 0.4442, 0),
                            ("cut 0.32", 0.4276, 0), ("cut 0.45", 0.3901, 0)]),
 ("Proposal source",       [("detector only", 0.4436, 1), ("+ semantic, w=0.3", 0.3637, 0),
                            ("+ semantic, w=0.5", 0.3450, 0), ("+ semantic, w=0.8", 0.3228, 0)]),
]

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
n = sum(len(v) for _, v in G) + len(G) - 1
fig, ax = plt.subplots(figsize=(7.0, 0.235 * (n + len(G)) + 0.75))

ax.axvspan(BASE - NOISE, BASE + NOISE, color="#4c72b0", alpha=0.14, zorder=0)
ax.axvline(BASE, color="#4c72b0", lw=1.0, ls="--", zorder=1)

y, ticks, labels, heads = 0.0, [], [], []
for gi, (grp, items) in enumerate(G):
    heads.append((grp, y))
    ticks.append(y); labels.append("")
    y -= 1
    for lab, pq, board in items:
        col = "#1f4e79" if abs(pq - BASE) <= NOISE else "#c44e52"
        ax.plot([BASE, pq], [y, y], color="0.85", lw=0.7, zorder=2)
        ax.scatter(pq, y, s=44, color=col if board else "white", edgecolor=col,
                   linewidth=1.4, zorder=4)
        ticks.append(y); labels.append(lab)
        y -= 1
    y -= 0.55

ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.8)
ax.set_ylim(y - 2.3, 0.9)
ax.set_xlim(0.305, 0.468)
ax.set_xlabel("Panoptic Quality on the held-out split", fontsize=9)
for grp, yy in heads:
    ax.text(0.311, yy, grp, ha="left", va="center", fontsize=8.4,
            fontweight="bold", color="#222222")

h = [plt.Line2D([], [], marker="o", ls="", mfc="#1f4e79", mec="#1f4e79", ms=6,
                label="also scored on the leaderboard (every one 0.36)"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#1f4e79", ms=6,
                label="local evaluation only"),
     plt.Rectangle((0, 0), 1, 1, fc="#4c72b0", alpha=0.14,
                   label=f"baseline $\\pm$ measured noise ({NOISE:.3f})")]
ax.legend(handles=h, frameon=False, fontsize=7.2, loc="lower left",
          bbox_to_anchor=(0.005, 0.005), handletextpad=0.6)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/interventions.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
