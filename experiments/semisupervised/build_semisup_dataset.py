"""Monta o dataset YOLO-seg semi-supervisionado: reais + pseudo-rotulados.

A validacao continua sendo APENAS as 35 imagens reais retidas -- nenhum pseudo-rotulo entra
nela, senao estariamos validando contra as proprias predicoes do modelo.
"""
import argparse, random, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pseudo", default="/data/marchezi/gong_pseudo")
ap.add_argument("--real", default=None, help="default: work/yolo_seg_dataset")
ap.add_argument("--out", default=None, help="default: work/yolo_seg_semisup")
ap.add_argument("--ratio", type=float, default=2.0,
                help="pseudo por imagem real. 2.0 = duas pseudo para cada real")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

root = Path(__file__).resolve().parents[1]
real = Path(a.real) if a.real else root / "work" / "yolo_seg_dataset"
out = Path(a.out) if a.out else root / "work" / "yolo_seg_semisup"
pseudo = Path(a.pseudo)

for split in ("train", "val"):
    for kind in ("images", "labels"):
        d = out / kind / split
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

n_real = 0
for split in ("train", "val"):
    for img in sorted((real / "images" / split).iterdir()):
        lab = real / "labels" / split / f"{img.stem}.txt"
        if not lab.exists(): continue
        (out / "images" / split / img.name).symlink_to(img.resolve())
        (out / "labels" / split / lab.name).symlink_to(lab.resolve())
        if split == "train": n_real += 1

cand = sorted(p for p in (pseudo / "labels").glob("*.txt"))
random.Random(a.seed).shuffle(cand)
want = int(n_real * a.ratio)
n_ps = 0
for lab in cand[:want]:
    img = pseudo / "images" / f"{lab.stem}.jpg"
    if not img.exists(): continue
    (out / "images" / "train" / img.name).symlink_to(img.resolve())
    (out / "labels" / "train" / lab.name).symlink_to(lab.resolve())
    n_ps += 1

(out / "data.yaml").write_text(
    f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
n_val = len(list((out / "images" / "val").iterdir()))
print(f"treino: {n_real} reais + {n_ps} pseudo = {n_real + n_ps}  (razao {n_ps/max(n_real,1):.2f})")
print(f"validacao: {n_val} imagens, TODAS reais")
print(f"data.yaml: {out / 'data.yaml'}")
