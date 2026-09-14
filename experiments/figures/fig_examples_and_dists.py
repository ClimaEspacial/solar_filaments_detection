"""Figuras: exemplos qualitativos e distribuicoes de sobreposicao dos pares casados.

Uma unica passada de inferencia sobre a particao retida alimenta as duas.
"""
import cv2, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner

CONF, MT = 0.20, 0.5          # ponto de operacao apos a correcao do limiar de mascara
FIGDIR = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures"

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                      conf_thres=CONF, iou_thres=0.5, imgsz=2048)

seen, uniq = set(), []
for i in val_ids:
    fn = d.images_by_id[i]["file_name"]
    if fn not in seen: seen.add(fn); uniq.append((i, fn))

ious, dices, store = [], [], {}
for img_id, fn in uniq:
    img = load_image(paths.train_img_dir / fn)
    lm, _ = refine_candidates(img, det.detect(img), unet, dev, crop_size=ccfg.crop_size,
        pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=MT,
        min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
    preds = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
    gts = [polygon_to_mask(a["segmentation"], *img.shape).astype(bool)
           for a in d.anns_by_image.get(img_id, [])]
    gts = [g for g in gts if g.any()]
    for g in gts:
        best_i, best_d = 0.0, 0.0
        for m in preds:
            inter = (g & m).sum(); u = (g | m).sum()
            if u and inter / u > best_i:
                best_i, best_d = inter / u, 2 * inter / (g.sum() + m.sum())
        if best_i > 0.5:
            ious.append(best_i); dices.append(best_d)
    store[fn] = (img, preds, gts)

ious, dices = np.array(ious), np.array(dices)
print(f"{len(ious)} pares casados | IoU medio {ious.mean():.3f} | Dice medio {dices.mean():.3f}")

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
for a_, v, nm, col in ((ax[0], ious, "IoU", "#4c72b0"), (ax[1], dices, "Dice", "#55a868")):
    a_.hist(v, bins=np.linspace(0.5 if nm == "IoU" else 0.6, 1.0, 26), color=col,
            alpha=0.85, edgecolor="white", lw=0.4)
    a_.axvline(v.mean(), color="#c44e52", ls="--", lw=1.1, label=f"mean {v.mean():.3f}")
    a_.set_xlabel(f"{nm} of matched pairs"); a_.set_ylabel("Count")
    a_.legend(frameon=False, fontsize=7.5)
fig.tight_layout(pad=0.5, w_pad=1.4)
fig.savefig(f"{FIGDIR}/distributions.pdf", bbox_inches="tight")
fig.savefig(f"{FIGDIR}/distributions.png", dpi=200, bbox_inches="tight")

# --- exemplos: as duas imagens com mais filamentos de referencia ---
best = sorted(store.items(), key=lambda kv: -len(kv[1][2]))[:2]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.7))
for a_, (fn, (img, preds, gts)) in zip(ax, best):
    a_.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    for g in gts:
        for c in cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a_.plot(*np.vstack([c[:, 0], c[:1, 0]]).T, color="#e24a33", lw=0.8, ls="--")
    for m in preds:
        for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a_.plot(*np.vstack([c[:, 0], c[:1, 0]]).T, color="#00b7c7", lw=0.9)
    a_.set_xticks([]); a_.set_yticks([])
    a_.set_title(f"{len(preds)} predicted / {len(gts)} reference", fontsize=8)
ax[0].plot([], [], color="#00b7c7", lw=1.2, label="prediction")
ax[0].plot([], [], color="#e24a33", lw=1.2, ls="--", label="reference")
ax[0].legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor="white")
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(f"{FIGDIR}/examples.pdf", bbox_inches="tight")
fig.savefig(f"{FIGDIR}/examples.png", dpi=200, bbox_inches="tight")
print("salvos: distributions.pdf, examples.pdf")
