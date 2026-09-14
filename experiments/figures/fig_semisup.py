"""Figura: o desfecho do auto-treino e uma funcao da densidade de pseudo-rotulo.

Mesmo arquivo, mesmo modelo, mesma metrica -- so muda quanto de cada imagem foi rotulado.
Abaixo de ~75% da densidade real o detector aprende a ficar calado; a 90% o colapso some,
mas nao ha ganho em densidade nenhuma.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (densidade inst/img, PQ, GT perdido, rotulo)
D = [(5.34, 0.4332, 0.366, "model\nagreement"),
     (6.92, 0.4303, 0.348, "+ completeness\nfilter"),
     (8.26, 0.4453, 0.209, "flip\ninvariance")]
BASE_D, BASE_PQ, BASE_MISS = 9.18, 0.4571, 0.206

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
d = np.array([x[0] for x in D])

for a_, vals, base, lab, col, fmt in (
        (ax[0], [x[1] for x in D], BASE_PQ, "Panoptic Quality", "#1f4e79", "{:.3f}"),
        (ax[1], [x[2] for x in D], BASE_MISS, "Ground-truth miss rate", "#c44e52", "{:.1%}")):
    a_.axhline(base, color="0.55", ls="--", lw=1.0)
    a_.text(5.4, base, " real labels only", fontsize=7, color="0.4", va="bottom")
    a_.plot(np.r_[d, BASE_D], np.r_[vals, base], "o-", color=col, ms=5.5, lw=1.4)
    a_.scatter([BASE_D], [base], s=70, facecolor="white", edgecolor=col, lw=1.6, zorder=5)
    a_.set_xlabel("Pseudo-label density (instances per image)")
    a_.set_ylabel(lab)
    a_.set_xlim(4.7, 9.9)

ax[1].axvspan(4.7, 7.6, color="#c44e52", alpha=0.07)
ax[1].text(6.1, 0.30, "detector learns\nto stay silent", ha="center", fontsize=7.2, color="#c44e52")
ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
for i, (x, _, y, t) in enumerate(D):
    ax[1].annotate(t, (x, y), textcoords="offset points", xytext=(0, -22 if i < 2 else 12),
                   ha="center", fontsize=6.5, color="0.35")
for a_, l in zip(ax, "ab"):
    a_.text(0.04, 0.95, f"({l})", transform=a_.transAxes, fontsize=9, fontweight="bold", va="top")
fig.tight_layout(pad=0.5, w_pad=1.5)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/semisup.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
