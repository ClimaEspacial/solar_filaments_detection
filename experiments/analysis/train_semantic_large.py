"""Segmentador semantico GRANDE e pre-treinado, no disco inteiro a 512.

Buraco na nossa ablacao: variamos capacidade no detector (9M vs 27.6M) e no refinador de
recorte (8.8M vs 15.4M), mas o ramo semantico foi sempre o Flat U-Net de 0.25M, porque so o
usavamos como fonte secundaria de propostas. Esta e a receita que um competidor publicou
reivindicando 0.57: smp.Unet com encoder ResNet34 do ImageNet, 512x512, alvo = uniao das
anotacoes, e instancias por componentes conexos. Aqui ela e treinada na NOSSA particao para
ser avaliada pela NOSSA metrica.
"""
import argparse, cv2, numpy as np, torch, torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset as TorchDataset
from solar_filaments.config import Paths
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", default=None)
a = ap.parse_args()

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
train_ids, val_ids = load_split(paths.split_path)
# particao por FOTOGRAFIA, nao por registro de anotador (o notebook original divide por
# registro, o que vaza a mesma foto para os dois lados quando ela tem varios anotadores)
tr_files = sorted({d.images_by_id[i]["file_name"] for i in train_ids})
va_files = sorted({d.images_by_id[i]["file_name"] for i in val_ids})
print(f"{len(tr_files)} fotografias de treino, {len(va_files)} de validacao")

class Sem(TorchDataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fn = self.files[i]
        img = load_image(paths.train_img_dir / fn)
        m = np.zeros(img.shape, bool)
        for c in merge_annotator_records(d, fn, "union"):
            m |= polygon_to_mask(c["segmentation"], *img.shape).astype(bool)
        o = self.tf(image=img[..., None], mask=m.astype(np.float32))
        return o["image"], o["mask"].unsqueeze(0).float()

train_tf = A.Compose([A.Resize(a.imgsz, a.imgsz), A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
                      A.RandomRotate90(p=.5),
                      A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=.5),
                      A.RandomBrightnessContrast(0.2, 0.2, p=.5),
                      A.Normalize(mean=(0.5,), std=(0.5,)), A.pytorch.ToTensorV2()])
val_tf = A.Compose([A.Resize(a.imgsz, a.imgsz), A.Normalize(mean=(0.5,), std=(0.5,)),
                    A.pytorch.ToTensorV2()])
tl = DataLoader(Sem(tr_files, train_tf), batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True)
vl = DataLoader(Sem(va_files, val_tf), batch_size=a.batch_size, num_workers=4)

model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet", in_channels=1, classes=1).to(dev)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parametros")
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
bce = nn.BCEWithLogitsLoss()

def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((2, 3)) + eps
    den = p.sum((2, 3)) + y.sum((2, 3)) + eps
    return 1 - (num / den).mean()

out = a.out or str(paths.ckpt_dir / f"semantic_{a.encoder}_{a.imgsz}.pt")
best = 0.0
for ep in range(1, a.epochs + 1):
    model.train(); tot = 0.0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        lo = model(x)
        loss = bce(lo, y) + dice_loss(lo, y)
        loss.backward(); opt.step(); tot += loss.item() * x.size(0)
    sched.step()
    model.eval(); inter = union = 0.0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(dev), y.to(dev)
            p = (torch.sigmoid(model(x)) > 0.5).float()
            inter += (p * y).sum().item(); union += p.sum().item() + y.sum().item()
    dice = 2 * inter / max(union, 1)
    if dice > best:
        best = dice; torch.save(model.state_dict(), out)
    print(f"epoch {ep:03d}/{a.epochs}  loss={tot/len(tl.dataset):.4f}  val_dice={dice:.4f}", flush=True)
print(f"melhor val dice: {best:.4f} -> {out}")
