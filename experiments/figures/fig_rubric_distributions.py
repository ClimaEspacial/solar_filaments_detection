"""Figuras exigidas explicitamente pelo rubrico quantitativo (70%) da competicao:
distribuicao de Dice, distribuicao de IoU, e distribuicao das relacoes um-para-muitos e
muitos-para-um entre referencia e predicao. Configuracao adotada."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solar_filaments.config import CropConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner

CONF, MT = 0.25, 0.5
OUT = "kaggle_comp_report_fi_segm_Marchezi_2026/figures"
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                      conf_thres=CONF, iou_thres=0.5, imgsz=2048)

# uma passada de inferencia por IMAGEM FISICA, guardada por nome de arquivo -- registros de
# anotador da mesma imagem nao sao adjacentes em val_ids, entao um cache posicional erra
by_file = {}
for fn in sorted({d.images_by_id[i]["file_name"] for i in val_ids}):
    img = load_image(paths.train_img_dir / fn)
    lm, _ = refine_candidates(img, det.detect(img), unet, dev, crop_size=ccfg.crop_size,
        pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=MT,
        min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
    by_file[fn] = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]

ious, dices, gt_deg, pr_deg = [], [], [], []
for img_id in val_ids:
    info = d.images_by_id[img_id]; fn = info["file_name"]
    gts = [m for m in (polygon_to_mask(a["segmentation"], info["height"], info["width"]).astype(bool)
                       for a in d.anns_by_image.get(img_id, [])) if m.any()]
    preds = by_file[fn]
    for g in gts:
        n_hit, best_i, best_d = 0, 0.0, 0.0
        for m in preds:
            inter = (g & m).sum()
            if not inter: continue
            n_hit += 1
            u = (g | m).sum()
            if inter / u > best_i: best_i, best_d = inter / u, 2 * inter / (g.sum() + m.sum())
        gt_deg.append(n_hit)
        if best_i > 0.5: ious.append(best_i); dices.append(best_d)
    for m in preds:
        pr_deg.append(sum(1 for g in gts if (g & m).any()))

ious, dices = np.array(ious), np.array(dices)
gt_deg, pr_deg = np.array(gt_deg), np.array(pr_deg)
print(f"{len(ious)} pares casados | IoU medio {ious.mean():.3f} | Dice medio {dices.mean():.3f}")
print(f"GT 1:0 {np.mean(gt_deg==0):.1%}  1:1 {np.mean(gt_deg==1):.1%}  1:2+ {np.mean(gt_deg>=2):.1%}")
print(f"pred 0:1 {np.mean(pr_deg==0):.1%}  1:1 {np.mean(pr_deg==1):.1%}  2+:1 {np.mean(pr_deg>=2):.1%}")

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
for a_, v, nm, col in ((ax[0], ious, "IoU", "#4c72b0"), (ax[1], dices, "Dice", "#55a868")):
    a_.hist(v, bins=np.linspace(v.min(), 1.0, 26), color=col, alpha=0.85, edgecolor="white", lw=0.4)
    a_.axvline(v.mean(), color="#c44e52", ls="--", lw=1.1, label=f"mean {v.mean():.3f}")
    a_.set_xlabel(f"{nm} of matched pairs"); a_.set_ylabel("Count"); a_.legend(frameon=False, fontsize=7)
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(f"{OUT}/distributions.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4.6, 2.3))
lab = ["1:0\nmissed", "1:1", "1:2+\nfragmented", "0:1\nspurious", "2+:1\nover-merged"]
val = [np.mean(gt_deg==0), np.mean(gt_deg==1), np.mean(gt_deg>=2),
       np.mean(pr_deg==0), np.mean(pr_deg>=2)]
col = ["#c44e52", "#4c72b0", "#dd8452", "#c44e52", "#dd8452"]
b = ax.bar(range(5), [100*v for v in val], color=col, alpha=0.85, width=0.66)
for i, v in enumerate(val): ax.text(i, 100*v + 1.5, f"{v:.1%}", ha="center", fontsize=7, color="0.35")
ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7)
ax.set_ylabel("Share of instances (%)"); ax.set_ylim(0, max(val)*118)
ax.axvline(2.5, color="0.7", lw=0.8, ls=":")
ax.text(1.0, max(val)*108, "per reference filament", ha="center", fontsize=6.8, color="0.4")
ax.text(3.5, max(val)*108, "per prediction", ha="center", fontsize=6.8, color="0.4")
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/m2n_relations.pdf", bbox_inches="tight")
print("salvos: distributions.pdf, m2n_relations.pdf")
