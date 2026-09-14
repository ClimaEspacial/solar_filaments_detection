"""Figura: fracao de anotadores que marcam cada filamento agrupado.

Usa a mesma regra de agrupamento do avaliador (IoU de mascara > 0.5), sobre as imagens
que carregam mais de uma anotacao independente.
"""
import collections
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solar_filaments.config import Paths
from solar_filaments.data.coco import Dataset, merge_annotator_records

p = Paths(); d = Dataset(p.train_ann_path)
multi = [fn for fn in {im["file_name"] for im in d.images_by_id.values()}
         if len(d.file_name_to_image_ids[fn]) > 1]
sup = [round(c["support"], 2) for fn in sorted(multi) for c in merge_annotator_records(d, fn, "union")]
c = collections.Counter(sup); n = len(sup)
print(f"{n} filamentos agrupados em {len(multi)} imagens multi-anotador")
for k in sorted(c): print(f"  suporte {k:.2f}: {c[k]:4d}  {c[k]/n:5.1%}")

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(figsize=(4.4, 2.5))
ks = sorted(c)
ax.bar(range(len(ks)), [100 * c[k] / n for k in ks], color="#4c72b0", alpha=0.85,
       width=0.66, edgecolor="#1f4e79", lw=0.7)
for i, k in enumerate(ks):
    ax.text(i, 100 * c[k] / n + 0.7, f"{c[k]}", ha="center", fontsize=7, color="0.4")
ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:.2f}" for k in ks])
ax.set_xlabel("Fraction of annotators marking the filament")
ax.set_ylabel("Clustered filaments (\\%)")
ax.set_ylim(0, max(100 * c[k] / n for k in ks) * 1.18)
fig.tight_layout(pad=0.4)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/annotator_agreement.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out, f"| n={n}")
