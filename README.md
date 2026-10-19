# GraviLens: A Physics-Informed Vision Transformer for Dark Matter Classification via Strong Gravitational Lensing

A hybrid physics-guided Swin vision transformer for strong gravitational lens
classification via the singular isothermal sphere (SIS) model.

## Overview

GraviLens injects an SIS-physics-informed encoder into a Swin-style transformer
backbone and is benchmarked against five baselines (Lensformer, LensPINN-Small,
ViT, CaiT, ViTSD) on simulated strong-lensing images. The physics branch
predicts the SIS lens parameters (Einstein radius and lens-center offset) and
reconstructs a deflection field to guide attention, improving accuracy at a
fraction of the parameter count.

## v9 benchmark results

Single Tesla T4, seed 42, 2-class labels, 1,500 test images. Test F1 and
throughput are as recorded in the final run (`n_classes=2`); parameter counts
were measured from the released checkpoints.

| Model | Params (M) | Test F1 | Throughput @ batch 64 (img/s) |
|---|---|---|---|
| **GraviLens** | 4.78 | 0.9853 | 2807 |
| Lensformer | 15.79 | 0.9747 | 898 |
| LensPINN-Small | 32.44 | 0.9627 | 974 |
| ViT | 9.53 | 0.8864 | 22847 |
| CaiT | 13.77 | 0.8323 | 18171 |
| ViTSD | 13.73 | 0.8967 | 16662 |

*LensPINN-Small* is the ablation label used in the recorded run; it was trained
as a smaller Lensformer (patch 16, embed 384, 2 blocks) — see the `REGISTRY` in
`GraviLens/gravilens_sis/main.py`.

## Dataset

10,000 train / 1,500 val / 1,500 test simulated 64×64 strong-lensing images
(1 channel). The released v9 checkpoints were trained with 2 output classes.
Data arrays are `.npy` files: `data_{train,val,test}.npy` (images) and
`labels_{train,val,test}.npy` (one-hot labels).

## Install

```bash
git clone https://github.com/mahoutei/GraviLens-SIS
cd GraviLens-SIS
pip install -r requirements.txt
```

## Usage

Train and evaluate the benchmark suite (CPU-friendly smoke run shown; real
training was done on a GPU):

```bash
cd GraviLens
python -m gravilens_sis.main --data-dir /path/to/data --models vit
```

`--help` lists all options (`--checkpoint-dir`, `--device`, `--seed`, model
selection). The final interactive notebook that produced the results lives in
`notebooks/` (originally run on
[Colab](https://colab.research.google.com/drive/1Fj823busXrl4wNvBGXcBSHF2GRw_POn0)).

## Structure

```
GraviLens/gravilens_sis/
├─ configs.py        # v9 hyperparameter configurations
├─ main.py           # CLI entry point
├─ utils.py
├─ data/             # datasets + SIS distortion maps
├─ components/       # SPT tokenizers, window attention, LSA blocks
├─ physics/          # SIS deflection / predictor / physics encoders
├─ models/           # GraviLens + 5 baselines
├─ stability/        # warmup-cosine, EMA, lookahead, clipping, early stop
├─ training/         # trial manager + training loops
└─ viz/              # manuscript-ready plots and tables
```

Trained weights are not in this repository (they live with the authors;
`*.pth` is git-ignored).

## Tests

```bash
python -m pytest tests -q
```

## Citation

Catiil, D. M. H. (2026). *GraviLens: A Physics-Informed Vision Transformer for
Dark Matter Classification via Strong Gravitational Lensing.* 28th SPVM
National Physics Conference (ICPAR 2026), Ateneo de Davao University.
See `CITATION.cff`.

## License

MIT — see `LICENSE`.
