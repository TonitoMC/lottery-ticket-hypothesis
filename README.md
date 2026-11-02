# Lottery Ticket Hypothesis Across Architectures

Lottery Ticket Hypothesis (LTH) across architectures (MLP, CNN, ResNet-18, ViT): do winning tickets exist in each one, and how far into training do you have to rewind before they appear? Final project for Deep Learning.

## Notebooks (`notebooks/`)

| Notebook | What it does | Dataset |
|---|---|---|
| `01_mlp_mnist.py` | LeNet-300-100, IMP + random-reinit control | MNIST |
| `02_cnn_cifar10.py` | Conv-4, per-layer pruning | CIFAR-10 |
| `02b_cnn_global.py` | Same as 02 with global pruning. Removes the 10.7% collapse (75.7% vs. 53.0%) | CIFAR-10 |
| `03_resnet18_cifar10.py` | ResNet-18 (CIFAR variant) + rewinding to epoch 1 | CIFAR-10 |
| `04_vit_cifar10.py` | Small ViT, 20 epochs + rewinding to epoch 2 | CIFAR-10 |
| `05_vit_longer.py` | ViT, 100 epochs, RandAugment / mixup / cutmix / label smoothing, rewinding to epoch 5 | CIFAR-10 |
| `05_comparison.py` | Reads `results/*.json`, compares all architectures | - |
| `06_inspection.py` | Opens the saved checkpoints: per-layer / per-head survival, mask shapes, per-class accuracy, attention, gradients | - |
| `07_informe.py` | The report (Spanish): method, results, inspection, discussion questions | - |
| `08_demo.py` | Interactive demo: pick model, sparsity and image (or upload one); dense vs. ticket vs. random | - |

`notebooks/lth.py` is the pruning code (masks, magnitude pruning, IMP loop with optional rewinding, random-reinit control, checkpoint saving). Each notebook defines its own model and training loop. Exported HTML copies sit next to each notebook.

## Results at a glance

| Run | Dense | Ticket (best / sparsest) | Random reinit at the sparsest point |
|---|---|---|---|
| MLP / MNIST | 97.7% | 98.2% / 97.2% at 3.5% of weights | 95.7% |
| CNN / CIFAR-10 | 73.0% | 77.2% / collapses at 10.7% (53.0%) | 61.2% |
| ResNet-18 / CIFAR-10 | 90.9% | rewind-1: 89.6% at 10.7%; from θ₀: 86.3% | 86.5% |
| ViT, 20 epochs | 78.9% | 82.6% at 11.6% | 71.5% |
| ViT, 100 epochs | 86.0% | 89.5% (rewind-5) at 27.5%; 88.1% from θ₀ at 11.6% | 78.0% at 11.6% |

One seed per run; differences of about 1 point are noise. See `notebooks/07_informe.py` for the discussion and its limits.

## Checkpoints

Each IMP round writes `checkpoints/<run>/<tag>_rNN_<pct>pct.pt` (`tag` = `ticket`, `late` for the rewind curve, `random`) with the weights, boolean masks, accuracy and the fraction of weights left; `base.pt` holds the dense, initial and rewind weights. Load with `lth.load_checkpoint(path)` and list with `lth.list_checkpoints(dir)`. Only a demo subset (~235 MB: every `base.pt`, plus ticket/random(/late) at the most informative levels: MLP 41/10.7/3.5%; CNN per-layer 32.8% (its peak) and 10.7% (collapse); CNN global 10.7%; ViT 11.6% and 4.9% (where rewinding starts to matter)) is in git. The rest is git-ignored (~1.1 GB) and regenerates by re-running the notebooks. Only runs `01_mlp`, `02_cnn` and `05_vit_longer` have them (03 and 04 predate the saving code).

### Demo (`notebooks/08_demo.py`)

```bash
cd notebooks && uv run marimo run 08_demo.py
```

It works from a fresh clone using only the checkpoints in git: pick a model (CNN per-layer, CNN global, ViT), a pruning level among those available, and a CIFAR-10 test image (or upload one); it shows the predictions of the dense net, the ticket, the rewinding ticket (ViT) and the random control, the ViT attention (descriptive only, see the report: it does not predict occlusion importance), and the full accuracy curves from `results/*.json`. The first run downloads CIFAR-10 into `notebooks/data` (internet needed). It runs on CPU.

### Repo hygiene

- Logs (`*.log`) are git-ignored; the experiment logs were deleted because every number is already in `results/*.json`.
- LaTeX build files are git-ignored; `report/informe.pdf` is committed on purpose.
- To regenerate the report figures that need all rounds (ViT curves, block-depth survival), re-run notebooks 01-05 to recreate the full checkpoints first: `uv run python report/make_figures.py`.

## Slides

`presentation/slides.html` (arrow keys). `python3 presentation/build.py` rebuilds it from `results/*.json`; `presentation/make_figs.py` regenerates the figures of slide 9 from the checkpoints.

## How to run

```bash
uv sync
uv run marimo edit notebooks/01_mlp_mnist.py     # any notebook
cd notebooks && uv run marimo export html 07_informe.py -o 07_informe.html
INSPECT_DIR=../checkpoints/05_vit_longer INSPECT_PCT=10 uv run marimo edit 06_inspection.py
```

Each training notebook saves its curves to `results/` when it finishes; run 01-05 before the comparison and the report. Sources for the hyperparameters and where we deviate from them: [`REFERENCES.md`](REFERENCES.md). What is left to do: [`PENDING.md`](PENDING.md).
