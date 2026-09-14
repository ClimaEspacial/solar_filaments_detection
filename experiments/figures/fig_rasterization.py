"""Figura: a convencao de rasterizacao de poligonos muda a mascara e portanto a metrica.

`cv2.fillPoly` e `pycocotools.frPyObjects` discordam sistematicamente na fronteira, e a
discordancia e relativamente maior para estruturas finas -- exatamente o regime dos
filamentos. Este script reproduz o painel de exemplo e a estatistica de concordancia.
"""
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
from solar_filaments.config import Paths
from solar_filaments.data.coco import Dataset, load_image, polygon_to_mask

CROP_IMAGE_INDEX, CROP_ANN_INDEX = 0, 0   # deterministico; a estatistica usa TODAS as anotacoes

def cv2_mask(seg, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(seg, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)], 1)
    return m.astype(bool)

p = Paths(); d = Dataset(p.train_ann_path)
ids = sorted(d.images_by_id)

# --- estatistica de concordancia sobre uma amostra fixa de filamentos ---
ious, areas, ratios, only_cv2, only_coco = [], [], [], [], []
for img_id in ids:
    info = d.images_by_id[img_id]
    for ann in d.anns_by_image.get(img_id, []):
        a = polygon_to_mask(ann["segmentation"], info["height"], info["width"]).astype(bool)
        b = cv2_mask(ann["segmentation"], info["height"], info["width"])
        u = (a | b).sum()
        if u and b.sum():
            ious.append((a & b).sum() / u); areas.append(int(a.sum()))
            ratios.append(a.sum() / b.sum())
            only_cv2.append((b & ~a).sum() / u); only_coco.append((a & ~b).sum() / u)
ious, areas = np.array(ious), np.array(areas)
med = np.median(areas)
ratios, only_cv2, only_coco = np.array(ratios), np.array(only_cv2), np.array(only_coco)
print(f"n={len(ious)}  IoU medio={ious.mean():.3f}")
print(f"  razao de area mediana (pycocotools/cv2) = {np.median(ratios):.3f}")
print(f"  area so no cv2       : {only_cv2.mean():.1%}")
print(f"  area so no pycocotools: {only_coco.mean():.1%}")
print(f"  filamentos abaixo da mediana de area: IoU medio={ious[areas < med].mean():.3f}")
print(f"  acima da mediana                    : IoU medio={ious[areas >= med].mean():.3f}")

# --- painel de exemplo ---
info = d.images_by_id[ids[CROP_IMAGE_INDEX]]
ann = d.anns_by_image[ids[CROP_IMAGE_INDEX]][CROP_ANN_INDEX]
img = load_image(p.train_img_dir / info["file_name"])
A = polygon_to_mask(ann["segmentation"], *img.shape).astype(bool)
B = cv2_mask(ann["segmentation"], *img.shape)
ys, xs = np.where(A | B)
pad = 25
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
crop, Ac, Bc = img[y0:y1, x0:x1], A[y0:y1, x0:x1], B[y0:y1, x0:x1]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw={"width_ratios": [1, 1, 1.15]})
ax[0].imshow(crop, cmap="gray"); ax[0].set_title("H$\\alpha$ crop", fontsize=8)
over = np.stack([crop] * 3, -1).astype(float) / max(crop.max(), 1)
over[Bc & ~Ac] = [0.85, 0.1, 0.1]      # so cv2
over[Ac] = [0.15, 0.3, 0.8]            # pycocotools
ax[1].imshow(over); ax[1].set_title("blue: pycocotools   red: cv2-only", fontsize=8)
for a_ in ax[:2]: a_.set_xticks([]); a_.set_yticks([])
ax[2].scatter(areas, ious, s=3.5, alpha=0.22, color="#4c72b0", edgecolor="none", rasterized=True)
ax[2].axhline(ious.mean(), color="#c44e52", ls="--", lw=1.1,
              label=f"mean {ious.mean():.3f}")
ax[2].set_xscale("log"); ax[2].set_xlabel("Reference filament area (px)")
ax[2].set_ylabel("IoU(cv2, pycocotools)"); ax[2].legend(frameon=False, fontsize=7, loc="lower right")
ax[2].set_title(f"agreement vs size (all $n = {len(ious)}$)", fontsize=8)
fig.tight_layout(pad=0.4, w_pad=1.0)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/rasterization.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
