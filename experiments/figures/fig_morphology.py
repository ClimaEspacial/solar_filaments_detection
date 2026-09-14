"""Figura exigida pelo rubrico qualitativo: morfologia aparente das segmentacoes em H-alfa.

A vista de disco inteiro nao resolve barbs -- as estruturas finas que o desafio destaca como
primeira dificuldade. Aqui, recortes ampliados de filamentos casados, escolhidos por
elongacao para mostrar o eixo e as ramificacoes.
"""
import cv2, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solar_filaments.config import CropConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner

CONF, MT, N = 0.25, 0.5, 6
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                      conf_thres=CONF, iou_thres=0.5, imgsz=2048)

cands = []
for fn in sorted({d.images_by_id[i]["file_name"] for i in val_ids}):
    img = load_image(paths.train_img_dir / fn)
    lm, _ = refine_candidates(img, det.detect(img), unet, dev, crop_size=ccfg.crop_size,
        pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=MT,
        min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
    preds = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
    info = d.images_by_id[d.file_name_to_image_ids[fn][0]]
    for a in d.anns_by_image[d.file_name_to_image_ids[fn][0]]:
        g = polygon_to_mask(a["segmentation"], info["height"], info["width"]).astype(bool)
        if not g.any(): continue
        best, bm = 0.0, None
        for m in preds:
            u = (g | m).sum()
            if u and (g & m).sum() / u > best: best, bm = (g & m).sum() / u, m
        if best > 0.5 and g.sum() > 900:
            ys, xs = np.where(g)
            cov = np.cov(np.vstack([xs, ys])); ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
            cands.append((elong, best, img, g, bm))

cands.sort(key=lambda t: -t[0])            # os mais alongados mostram melhor eixo e barbs
sel = cands[:N]
print(f"{len(cands)} filamentos casados; usando os {N} mais alongados "
      f"(elongacao {sel[-1][0]:.1f}-{sel[0][0]:.1f})")

plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
for ax, (elong, iou, img, g, m) in zip(axes.ravel(), sel):
    ys, xs = np.where(g | m)
    pad = 30
    y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, img.shape[0])
    x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, img.shape[1])
    sub = img[y0:y1, x0:x1]
    ax.imshow(sub, cmap="gray", vmin=np.percentile(sub, 2), vmax=np.percentile(sub, 98))
    ax.contour(g[y0:y1, x0:x1], [0.5], colors="#e24a33", linewidths=1.0, linestyles="dashed")
    ax.contour(m[y0:y1, x0:x1], [0.5], colors="#00b7c7", linewidths=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"IoU {iou:.2f}", fontsize=7, pad=2)
axes[0, 0].plot([], [], color="#00b7c7", lw=1.3, label="prediction")
axes[0, 0].plot([], [], color="#e24a33", lw=1.3, ls="--", label="reference")
axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left", labelcolor="white")
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
out = "kaggle_comp_report_fi_segm_Marchezi_2026/figures/morphology.pdf"
fig.savefig(out, bbox_inches="tight"); print("salvo:", out)
