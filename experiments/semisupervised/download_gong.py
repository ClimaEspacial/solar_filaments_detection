"""Baixa imagens H-alfa nao rotuladas do arquivo do GONG, para treino semi-supervisionado.

Amostra dias espalhados por 2011-2022 (a cobertura do MAGFiLO) e, em cada dia, algumas
imagens de estacoes aleatorias. Exclui a estacao Z (ausente do MAGFiLO) e qualquer imagem
que ja esteja no MAGFiLO -- em particular as de TESTE, que jamais podem entrar no treino.
Educado com o servidor: 4 conexoes, pausa entre listagens, e nunca segue links da pagina.
"""
import concurrent.futures as cf, random, re, sys, time
from pathlib import Path
import urllib.request

BASE = "https://gong2.nso.edu/HA/hag"
OUT = Path("/data/marchezi/gong_unlabeled")
DATA = Path("/data/marchezi/filament_segmentation_2026/MAGFiLO_1.0_Kaggle_2026")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
PER_DAY = 6
STATIONS = set("BCLMTU")

OUT.mkdir(parents=True, exist_ok=True)
known = {p.stem for d in ("train/train_images", "test/test_images") for p in (DATA / d).glob("*.jpeg")}
print(f"{len(known)} imagens do MAGFiLO serao excluidas (treino + TESTE)")

def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return r.read()
        except Exception:
            if k == tries - 1: return None
            time.sleep(2 * (k + 1))

days = []
for year in range(2011, 2023):
    for month in range(1, 13):
        days.append(f"{year}{month:02d}")
random.seed(0)
months = days                      # todos os meses, alguns dias em cada
per_month = max(1, TARGET // (len(months) * PER_DAY) + 1)
print(f"{len(months)} meses, ~{per_month} dias por mes, ate {PER_DAY} imagens por dia")

def day_urls(ym):
    body = get(f"{BASE}/{ym}/")
    if not body: return []
    ds = re.findall(rb'href="(\d{8})/"', body)
    if not ds: return []
    rng = random.Random(ym)
    out = []
    for day in rng.sample([d.decode() for d in ds], min(per_month, len(ds))):
        b2 = get(f"{BASE}/{ym}/{day}/")
        if not b2: continue
        files = [f.decode() for f in re.findall(rb'href="(\d{14}[A-Z]h\.jpg)"', b2)]
        files = [f for f in files if f[14] in STATIONS and f[:-4] not in known]
        out += [f"{BASE}/{ym}/{day}/{f}" for f in rng.sample(files, min(PER_DAY, len(files)))]
        time.sleep(0.3)
    return out

urls = []
with cf.ThreadPoolExecutor(4) as ex:
    for got in ex.map(day_urls, months):
        urls += got
        print(f"\r  URLs coletadas: {len(urls)}", end="", flush=True)
random.shuffle(urls); urls = urls[:TARGET]
print(f"\n{len(urls)} URLs para baixar")

def fetch(u):
    dst = OUT / Path(u).name
    if dst.exists(): return True
    b = get(u)
    if b and len(b) > 50_000:
        dst.write_bytes(b); return True
    return False

ok = 0
with cf.ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(fetch, urls), 1):
        ok += bool(r)
        if i % 100 == 0: print(f"\r  baixadas {ok}/{i}", end="", flush=True)
print(f"\nconcluido: {ok} imagens em {OUT}")
