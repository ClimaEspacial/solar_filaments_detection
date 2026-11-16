"""Figura: os falsos positivos sao filamentos nao anotados ou lixo?

(a) recortes das predicoes sem nenhuma sobreposicao com o GT, para inspecao
visual; (b) contraste e (c) alongamento, comparados aos verdadeiros positivos.
Se os FP tem a assinatura morfologica e fotometrica de filamentos, a remocao
deles nao e um problema de modelagem e sim o teto da anotacao.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("experiments/data/fp_stats.json"))
crops = np.load("experiments/data/fp_crops.npy", allow_pickle=True)
tp = [r for r in rows if r["kind"] == "TP"]
fp = [r for r in rows if r["kind"] == "FP0"]
g = lambda L, k: np.array([x[k] for x in L], float)

fig = plt.figure(figsize=(13.5, 6.4))
gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1.25], hspace=.35, wspace=.15)
for i in range(16):
    ax = fig.add_subplot(gs[i // 8, i % 8])
    ax.set_xticks([]); ax.set_yticks([])
    if i < len(crops):
        c = crops[i]
        ax.imshow(c, cmap="gray", vmin=np.percentile(c, 2), vmax=np.percentile(c, 98))
    else:
        ax.axis("off")
fig.text(.5, .955, "(a) predições sem nenhuma sobreposição com o ground truth",
         ha="center", fontsize=11)

ax = fig.add_subplot(gs[2, 0:3])
b = np.linspace(0, .6, 24)
ax.hist(g(tp, "rank"), bins=b, alpha=.6, color="#777", density=True, label=f"verdadeiros (n={len(tp)})")
ax.hist(g(fp, "rank"), bins=b, alpha=.6, color="#b2182b", density=True, label=f"falsos (n={len(fp)})")
ax.axvline(.5, ls=":", c="k", lw=1)
ax.set_xlabel("ranque de escuridão"); ax.set_ylabel("densidade")
ax.legend(fontsize=8, frameon=False); ax.set_title("(b) contraste", fontsize=10, loc="left")

ax = fig.add_subplot(gs[2, 3:6])
for L, lab, c in ((tp, "verdadeiros", "#777"), (fp, "falsos", "#b2182b")):
    ax.scatter(g(L, "width"), g(L, "length"), s=12, alpha=.6, c=c, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("largura (px)"); ax.set_ylabel("comprimento (px)")
ax.legend(fontsize=8, frameon=False); ax.set_title("(c) morfologia", fontsize=10, loc="left")

ax = fig.add_subplot(gs[2, 6:8])
ax.hist(g(tp, "r_frac"), bins=np.linspace(0, 1, 18), alpha=.6, color="#777", density=True)
ax.hist(g(fp, "r_frac"), bins=np.linspace(0, 1, 18), alpha=.6, color="#b2182b", density=True)
ax.set_xlabel("distância do centro / raio"); ax.set_title("(d) posição", fontsize=10, loc="left")
for a in fig.axes[16:]: a.spines[["top", "right"]].set_visible(False)
fig.savefig("experiments/figures/fig_false_positives.png", dpi=140, bbox_inches="tight")
fig.savefig("experiments/figures/fig_false_positives.pdf", bbox_inches="tight")
print("-> experiments/figures/fig_false_positives.{png,pdf}")
print(f"\n{'':14}{'ranque':>9}{'largura':>9}{'compr':>9}{'area':>9}{'r/R':>7}")
for L, lab in ((tp, "verdadeiros"), (fp, "falsos")):
    print(f"{lab:<14}{np.median(g(L,'rank')):>9.3f}{np.median(g(L,'width')):>9.1f}"
          f"{np.median(g(L,'length')):>9.1f}{np.median(g(L,'area')):>9.0f}{np.median(g(L,'r_frac')):>7.2f}")
