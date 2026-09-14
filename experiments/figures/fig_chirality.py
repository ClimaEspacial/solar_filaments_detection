"""Figura: regra hemisferica de quiralidade recuperada das anotacoes do MAGFiLO."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

arr = np.load("experiments/data/chirality.npy")   # (lat, category_id 1=Left 2=Right, ano)
lat, cat, yr = arr[:, 0], arr[:, 1], arr[:, 2]

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})

# --- (a) fracao dextral vs latitude, com erro binomial ---
edges = np.arange(-45, 50, 7.5)
c, f, e, n = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = cat[(lat >= lo) & (lat < hi)]
    if len(s) >= 30:
        k = (s == 2).sum(); p = k / len(s)
        c.append((lo + hi) / 2); f.append(p); n.append(len(s))
        e.append(np.sqrt(p * (1 - p) / len(s)))
c, f, e = np.array(c), np.array(f), np.array(e)
ax[0].axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=1)
ax[0].axvline(0, color="0.6", lw=0.8, ls=":", zorder=1)
ax[0].errorbar(c, f, yerr=e, fmt="o-", ms=4, lw=1.3, color="#1f4e79",
               ecolor="#1f4e79", capsize=2, zorder=3)
ax[0].fill_between([-50, 0], 0, 1, color="#c44e52", alpha=0.07, zorder=0)
ax[0].fill_between([0, 50], 0, 1, color="#4c72b0", alpha=0.07, zorder=0)
ax[0].text(-24, 0.80, "southern\nsinistral", ha="center", fontsize=8, color="#c44e52")
ax[0].text(26, 0.24, "northern\ndextral", ha="center", fontsize=8, color="#4c72b0")
ax[0].set_xlabel("Heliographic latitude (deg)")
ax[0].set_ylabel("Fraction dextral (right-bearing)")
ax[0].set_xlim(-48, 48); ax[0].set_ylim(0.15, 0.88)

# --- (b) contagens por latitude e quiralidade ---
b = np.arange(-50, 55, 5)
hR, _ = np.histogram(lat[cat == 2], bins=b); hL, _ = np.histogram(lat[cat == 1], bins=b)
ctr = (b[:-1] + b[1:]) / 2
ax[1].barh(ctr, hR, height=4.6, color="#4c72b0", alpha=0.85, label="dextral (right)")
ax[1].barh(ctr, -hL, height=4.6, color="#c44e52", alpha=0.85, label="sinistral (left)")
ax[1].axvline(0, color="0.3", lw=0.8)
ax[1].set_xticks([-300, -150, 0, 150, 300]); ax[1].set_xticklabels(["300", "150", "0", "150", "300"])
ax[1].axhline(0, color="0.6", lw=0.8, ls=":")
ax[1].set_ylabel("Heliographic latitude (deg)")
ax[1].set_xlabel("Annotated filaments  (sinistral | dextral)")
ax[1].set_ylim(-50, 50); ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")

for a_, l in zip(ax, "ab"):
    a_.text(0.03, 0.95, f"({l})", transform=a_.transAxes, fontsize=9, fontweight="bold", va="top")
fig.tight_layout(pad=0.4)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/chirality.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

N, S = cat[lat > 0], cat[lat < 0]
print(f"figura salva: {out}")
print(f"N: dextral {(N==2).mean():.1%} (n={len(N)})   S: sinistral {(S==1).mean():.1%} (n={len(S)})")
print(f"anos cobertos: {int(yr.min())}-{int(yr.max())}")
