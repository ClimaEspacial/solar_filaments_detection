#!/usr/bin/env bash
# Regenera todas as 9 figuras do artigo a partir dos dados e dos pesos treinados.
# Uso: experiments/figures/make_figures.sh   (a partir da raiz do repositorio)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

echo "== sem GPU: so anotacoes =="
$PY experiments/figures/fig_pipeline.py              # esquema, sem dados
$PY experiments/figures/fig_annotator_agreement.py   # concordancia entre anotadores (3254 clusters)
$PY experiments/figures/fig_rasterization.py         # cv2 vs pycocotools (todas as 8199 anotacoes)

echo "== quiralidade: precisa de chirality.npy (gerado abaixo se ausente) =="
[ -f experiments/data/chirality.npy ] || $PY - <<'PYX'
import sys, json, re, numpy as np; sys.path.insert(0, "experiments/figures")
from solar_angles import julian_day, sun_P_B0, heliographic_lat
from solar_filaments.config import Paths
raw = json.load(open(Paths().train_ann_path))
imgs = {im["id"]: im for im in raw["images"]}
CY, CX, R = 1024.0, 1024.0, 898.0   # geometria mediana do disco no arquivo
rows = []
for a in raw["annotations"]:
    im = imgs.get(a["image_id"])
    if im is None or a["category_id"] not in (1, 2): continue
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", im["file_name"])
    if not m: continue
    y_, mo, d_, hh, mi, ss = map(int, m.groups())
    P, B0 = sun_P_B0(julian_day(y_, mo, d_, hh, mi, ss))
    x, y, w, h = a["bbox"]
    lat = heliographic_lat((x + w/2) - CX, CY - (y + h/2), R, P, B0)
    if not np.isnan(lat): rows.append((lat, a["category_id"], y_))
np.save("experiments/data/chirality.npy", np.array(rows))
print(f"chirality.npy: {len(rows)} filamentos")
PYX
$PY experiments/figures/fig_chirality.py

echo "== com GPU: precisa dos pesos treinados =="
$PY experiments/figures/fig_mask_threshold.py        # varredura do limiar de mascara
$PY experiments/figures/fig_examples_and_dists.py    # exemplos + distribuicoes dos pares casados
$PY experiments/figures/fig_failures.py              # anatomia das falhas (valores medidos, embutidos)
$PY experiments/figures/fig_interventions.py         # panorama das intervencoes (valores medidos, embutidos)
$PY experiments/figures/fig_semisup.py           # dose-resposta do auto-treino (valores medidos, embutidos)
echo "== pronto: 9 figuras em Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/ =="
