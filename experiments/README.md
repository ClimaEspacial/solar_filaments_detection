# Experiments

Everything outside the installable package: the analyses that produced the reported
numbers, the figure scripts, the semi-supervised study, and the orchestration used for
the long training runs. All paths are relative to the repository root, so run these from
there (`python experiments/figures/fig_chirality.py`, not from inside the folder).

Outputs — weights, logs, datasets, submission CSVs — are written to `work/`, which is not
tracked. Only code and small derived arrays live here.

## Layout

| folder | what it holds |
|---|---|
| `figures/` | one script per figure in the report and paper, plus `make_figures.sh` to regenerate all ten |
| `analysis/` | the diagnostics: failure anatomy, proposal complementarity, re-scorer sweeps, refiner comparison |
| `semisupervised/` | GONG archive download, pseudo-labelling (two selection rules), dataset assembly, run chains |
| `pipelines/` | end-to-end shell chains for data prep, training and submission generation |
| `data/` | small derived arrays checked in so figures regenerate without a GPU pass |

The package itself is tested separately (`pytest` from the repository root, 40 tests).
Scripts here are analysis code: they are run once, their outputs are reported, and they are
kept so the numbers can be re-derived rather than trusted.

## Reproducing the figures

```bash
experiments/figures/make_figures.sh
```

Four of the ten need a GPU and trained weights; the rest run from the annotations alone.
`figures/solar_angles.py` implements the solar position angle *P* and the heliographic
latitude of disk centre *B*₀ (Meeus 1998, ch. 29), validated against reference values:
*B*₀ = +7.24° on 6 September against a theoretical maximum of +7.25°, −7.25° on 6 March,
and *P* crossing zero in early January and early July.

## The semi-supervised study

```bash
python experiments/semisupervised/download_gong.py 2500     # ~1.4 GB, excludes every MAGFiLO image
python experiments/semisupervised/pseudo_label.py           # agreement between detector and semantic branch
python experiments/semisupervised/pseudo_flipconsistent.py  # invariance under horizontal mirroring
python experiments/semisupervised/build_semisup_dataset.py --pseudo <dir> --out <dir>
filament train-yolo --seg --imgsz 2048 --data <dir>/data.yaml --name <run>
```

The download step verifies that no MAGFiLO image — in particular none from the withheld
test partition — enters the unlabelled pool. That check is not optional: a single leaked
test image would invalidate every number downstream.
