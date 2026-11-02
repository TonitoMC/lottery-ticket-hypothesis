# Lottery Ticket Hypothesis across architectures

Final project for Deep Learning (CC3092). We test the Lottery Ticket Hypothesis (LTH) on four architectures of increasing complexity (MLP, CNN, ResNet-20 and a small ViT) and ask three questions:

1. Do winning tickets exist in each architecture?
2. Does the point you rewind to (and warmup) change with complexity?
3. Does it matter whether you prune layer by layer or globally?

**Links:** [Report (PDF, Spanish)](report/informe.pdf) · [Slides](presentation/slides.html) ([PDF](presentation/slides.pdf)) · Video: ENLACE_YOUTUBE · Repo: https://github.com/TonitoMC/lottery-ticket-hypothesis

## Main findings (one seed per run; differences of about 1 point are noise)

- **Tickets exist in all four architectures.** In the ResNet only with rewinding to epoch 1; from the initial weights it does no better than the random-reinit control (86.3% vs. 86.5%).
- **Rewinding depends on the architecture and the training recipe.** The ResNet (no warmup) needs it; the MLP, the CNN and the 20-epoch ViT do not; the 100-epoch ViT (with warmup) only gains below about 8% of the weights. Warmup was never varied, so "warmup does the job of rewinding" is compatible with the data but not tested.
- **How you prune changes the result.** On the CNN, per-layer pruning collapses at 10.7% of the weights (53.0% vs. 61.2% random), global pruning does not (75.7% vs. 69.4%). Small layers have the largest weights and per-layer pruning makes them lose as much as the huge first linear layer.
- **The ViT is where we went deepest.** The 20-epoch baseline was undertrained (78.9%), so we repeated it with 100 epochs (86.0%). Raw attention does not predict which patches matter (rank correlation with occlusion about 0), so attention maps are only descriptive.

| Run | Dense | Best ticket | Ticket at the sparsest point | Rewind at the sparsest point | Random reinit at the sparsest point |
|---|---|---|---|---|---|
| MLP / MNIST | 97.7% | 98.2% (13.4% of weights) | 97.2% (3.5%) | - | 95.7% |
| CNN, per-layer pruning | 73.0% | 77.2% (32.8%) | 53.0% (10.7%) | - | 61.2% |
| CNN, global pruning | 74.0% | 76.9% (26.2%) | 75.7% (10.7%) | - | 69.4% |
| ResNet-20 | 90.9% | 91.3% (80%) | 86.3% (10.7%) | 89.6% | 86.5% |
| ViT, 20 epochs | 78.9% | 82.6% (42.3%) | 80.0% (3.2%) | 79.6% | 65.9% |
| ViT, 100 epochs | 86.0% | 88.9% (27.5%) | 77.3% (3.2%) | 78.4% | 68.5% |

## Repository layout

```
notebooks/      working marimo notebooks (.py) and lth.py, the pruning code
rendered/       the same notebooks exported to HTML (open them in a browser, no setup)
report/         LaTeX report (informe.tex, informe.pdf), make_figures.py, bibliography
presentation/   slides (template.html + build.py -> slides.html) and slide figure scripts
results/        per-experiment curves (JSON) that the report and slides read
checkpoints/    saved weights and masks (a demo subset is in git, see below)
```

| Notebook | What it does | Dataset |
|---|---|---|
| `01_mlp_mnist.py` | LeNet-300-100, IMP + random-reinit control | MNIST |
| `02_cnn_cifar10.py` | Conv-4, per-layer pruning | CIFAR-10 |
| `02b_cnn_global.py` | Same as 02 with global pruning | CIFAR-10 |
| `03_resnet18_cifar10.py` | 20-layer CIFAR ResNet (called ResNet-18 in the LTH paper), rewinding to epoch 1 | CIFAR-10 |
| `04_vit_cifar10.py` | Small ViT, 20 epochs, rewinding to epoch 2 | CIFAR-10 |
| `05_vit_longer.py` | ViT, 100 epochs, stronger augmentation, rewinding to epoch 5 (main ViT experiment) | CIFAR-10 |
| `06_comparison.py` | Reads `results/*.json` and compares all architectures | - |
| `07_inspection.py` | Opens the saved checkpoints: per-layer survival, masks, per-class accuracy, attention, gradients | - |
| `08_demo.py` | Interactive demo (see below) | - |

**About `rendered/`.** Each file is a notebook exported with its outputs. `01_mlp_mnist` and `02_cnn_cifar10` show the first run; `results/*.json`, the checkpoints and the report use the later rerun that also saved checkpoints, so a few numbers differ slightly (for example the per-layer CNN at 10.7% of the weights: 49.7% in the first run, 53.0% in the rerun; the report mentions both). `05_vit_longer` has no HTML because exporting re-runs the 5-hour training; its results are in `results/vit_longer*.json` and appear in `06_comparison` and `07_inspection`.

`notebooks/lth.py` holds the pruning code: masks, magnitude pruning, the IMP loop with optional rewinding, the random-reinit control and checkpoint saving. Each notebook defines its own model and training loop.

## Running it

```bash
uv sync

# any notebook, editable
uv run marimo edit notebooks/01_mlp_mnist.py

# export a notebook to rendered/ (this re-runs every cell)
cd notebooks && uv run marimo export html 06_comparison.py -o ../rendered/06_comparison.html

# the inspection notebook needs a checkpoint folder and a sparsity level
INSPECT_DIR=../checkpoints/05_vit_longer INSPECT_PCT=10 uv run marimo export html 07_inspection.py -o ../rendered/07_inspection.html
```

Each training notebook writes its curves to `results/` when it finishes; run 01-05 before the comparison and the report.

### Demo

```bash
cd notebooks && uv run marimo run 08_demo.py
```

Pick a model (CNN per-layer, CNN global or ViT), a pruning level and a CIFAR-10 test image (or upload your own). A filter jumps straight to images where the ticket is right and the random control is wrong (or where the ticket fails). It shows the predictions of the dense network, the ticket, the rewinding ticket (ViT) and the random control, the ViT attention (descriptive only) and the accuracy curves of the whole experiment. It works from a fresh clone with the checkpoints in git and runs on CPU; the first run downloads CIFAR-10 into `notebooks/data`.

### Report, figures and slides

```bash
uv run python report/make_figures.py          # figures and tables of the report
cd report && latexmk -pdf informe.tex          # report/informe.pdf
python3 presentation/build.py                  # presentation/slides.html (arrow keys)
uv run python report/make_figures.py --dark    # architecture diagrams for the slides
```

## Checkpoints

Each IMP round writes `checkpoints/<run>/<tag>_rNN_<pct>pct.pt` (`tag` = `ticket`, `late` for the rewinding curve, `random`) with the weights, boolean masks, accuracy and fraction of weights left; `base.pt` holds the dense, initial and rewind weights. Load with `lth.load_checkpoint(path)`, list with `lth.list_checkpoints(dir)`.

Only a demo subset (~235 MB) is in git: every `base.pt`, plus ticket / random (/ rewinding) at the most informative levels (MLP 41%, 10.7%, 3.5%; CNN per-layer 32.8% and 10.7%; CNN global 10.7%; ViT 11.6% and 4.9%). The full set (~1.1 GB) is git-ignored and regenerates by re-running the notebooks. Runs 01, 02, 02b and 05 have checkpoints (03 and 04 predate the saving code), and regenerating the report figures that use every round (ViT curves, block-depth survival) needs the full set.

## Use of AI

The author wrote the four models, the training and evaluation loops and `lth.py` by hand. Claude Code (Anthropic) was then used to wire `lth.py` into the other notebooks, run the experiments, add checkpoint saving, generate figures and tables, and draft text. The report's section 5.2 describes what worked, what failed and how each output was checked. Every number in the report comes from `results/*.json` or the checkpoints.

## References

1. Frankle & Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR 2019. https://arxiv.org/abs/1803.03635
2. Frankle, Dziugaite, Roy & Carbin, *Stabilizing the Lottery Ticket Hypothesis*, 2019. https://arxiv.org/abs/1903.01611
3. Renda, Frankle & Carbin, *Comparing Rewinding and Fine-tuning in Neural Network Pruning*, ICLR 2020. https://arxiv.org/abs/2003.02389
4. Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021. https://arxiv.org/abs/2012.12877
5. Shen et al., *Data Level Lottery Ticket Hypothesis for Vision Transformers*, IJCAI 2023. https://arxiv.org/abs/2211.01484
6. Jain & Wallace, *Attention is not Explanation*, NAACL 2019. https://arxiv.org/abs/1902.10186

Where our setup departs from these sources (epochs per round, pruning rates, ViT learning rate and warmup, etc.) is listed in Appendix A of the report.
