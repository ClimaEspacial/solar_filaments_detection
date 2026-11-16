"""Figura: AIA 304 acrescenta informacao sobre os filamentos que perdemos?

Painel (a) distribuicao do ranque de escuridao nos dois canais, separando
filamentos detectados e perdidos. Painel (b) recortes de filamentos perdidos
nos dois canais, co-registrados. Painel (c) area vs ranque.
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1] if len(sys.argv) > 1 else "experiments/data"
rows = json.load(open(f"{S}/premise2.json"))
dts = json.load(open(f"{S}/dts.json"))
rows = [r for r in rows if dts.get(r["file"], 0) < 1200
        and all(np.isfinite(r[k]) for k in ("r_ha", "r_304"))]
iou = np.array([r["iou"] for r in rows]); area = np.array([r["area"] for r in rows])
rha = np.array([r["r_ha"] for r in rows]); r304 = np.array([r["r_304"] for r in rows])
det, miss = iou > 0.5, iou <= 0.1

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
b = np.linspace(0, 0.8, 22)
for a_, lab, col in ((rha, r"H$\alpha$ (GONG)", "#b2182b"), (r304, r"304 Å (SDO/AIA)", "#2166ac")):
    ax[0].hist(a_[miss], bins=b, alpha=.55, color=col, label=f"{lab}", density=True)
ax[0].axvline(.5, ls=":", c="k", lw=1)
ax[0].text(.5, ax[0].get_ylim()[1]*.95, " indistinguishable", fontsize=8, va="top")
ax[0].set_xlabel("darkness rank (0 = distinct, 0.5 = indistinguishable)")
ax[0].set_ylabel("density"); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("(a) filaments MISSED by the model", fontsize=10, loc="left")

w = 0.35; x = np.arange(2)
for i, (m, lab) in enumerate(((det, "detected"), (miss, "missed"))):
    ax[1].bar(x + (i - .5) * w, [np.median(rha[m]), np.median(r304[m])], w,
              label=lab, color=["#777", "#d6604d"][i])
ax[1].set_xticks(x); ax[1].set_xticklabels([r"H$\alpha$", "304 Å"])
ax[1].axhline(.5, ls=":", c="k", lw=1)
ax[1].set_ylabel("median darkness rank"); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_title("(b) contrast by channel", fontsize=10, loc="left")

ax[2].scatter(area[det], rha[det], s=14, c="#777", label=r"detected, H$\alpha$")
ax[2].scatter(area[miss], rha[miss], s=22, c="#b2182b", label=r"missed, H$\alpha$")
ax[2].scatter(area[miss], r304[miss], s=22, c="#2166ac", marker="^", label="missed, 304 Å")
ax[2].set_xscale("log"); ax[2].axhline(.5, ls=":", c="k", lw=1)
ax[2].set_xlabel("filament area (px)"); ax[2].set_ylabel("darkness rank")
ax[2].legend(fontsize=7, frameon=False)
ax[2].set_title("(c) size is what fails, not contrast", fontsize=10, loc="left")
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("experiments/figures/fig_aia304.pdf")
fig.savefig("experiments/figures/fig_aia304.png", dpi=150)
print("-> experiments/figures/fig_aia304.{pdf,png}")
