# Lottery Ticket Hypothesis across architectures

Final project for Deep Learning (CC3092).

The Lottery Ticket Hypothesis says a big network hides a much smaller subnetwork that, trained from the same starting weights, does about as well as the full thing. I tried it on four networks of growing complexity (an MLP, a small CNN, a ResNet-20 and a small ViT) and wanted to know three things: whether those subnetworks ("tickets") show up in each one, whether you have to rewind the weights to a later point in training as the network gets more complex, and whether pruning layer by layer or globally changes the result.

- [Report](report/informe.pdf) (in Spanish)
- [Slides](presentation/slides.html) / [PDF](presentation/slides.pdf)
- [Video](https://youtu.be/ow74cZd0Ldg)

## What I found

One seed per run, so differences of about 1 point are just noise.

Tickets exist in all four networks, but the ResNet only gives you one if you rewind to epoch 1. From the initial weights it does no better than the control, which keeps the same mask but uses fresh random weights (86.3% vs. 86.5%).

Whether you need to rewind depends on the network and on how it was trained. The ResNet (no warmup) needs it. The MLP, the CNN and the 20-epoch ViT don't. The 100-epoch ViT only benefits below roughly 8% of the weights. I never varied the warmup, so "warmup does the job of rewinding" fits the data but isn't something I tested.

The way you prune matters. On the CNN, pruning each layer by the same fraction makes the ticket collapse at 10.7% of the weights (53.0% vs. 61.2% for the control). Pruning globally doesn't (75.7% vs. 69.4%). The small layers have the biggest weights, and per-layer pruning strips them as hard as the huge first linear layer.

The ViT is where I dug deepest. My first 20-epoch run was undertrained (78.9%), so I redid it with 100 epochs (86.0%). I also checked whether its attention maps explain anything: they don't predict which patches matter, so I only use them as a description.

| Run | Dense | Best ticket | Ticket, sparsest | Rewind, sparsest | Random, sparsest |
|---|---|---|---|---|---|
| MLP / MNIST | 97.7% | 98.2% (13.4% of weights) | 97.2% (3.5%) | - | 95.7% |
| CNN, per-layer | 73.0% | 77.2% (32.8%) | 53.0% (10.7%) | - | 61.2% |
| CNN, global | 74.0% | 76.9% (26.2%) | 75.7% (10.7%) | - | 69.4% |
| ResNet-20 | 90.9% | 91.3% (80%) | 86.3% (10.7%) | 89.6% | 86.5% |
| ViT, 20 epochs | 78.9% | 82.6% (42.3%) | 80.0% (3.2%) | 79.6% | 65.9% |
| ViT, 100 epochs | 86.0% | 88.9% (27.5%) | 77.3% (3.2%) | 78.4% | 68.5% |

## What's in the repo

```
notebooks/      the marimo notebooks (.py) and lth.py, the pruning code
rendered/       the same notebooks exported to HTML, so you can read them without running anything
report/         the LaTeX report and the script that makes its figures
presentation/   the slides
results/        the curves from each experiment (JSON); the report and slides read these
checkpoints/    saved weights and masks (only a subset is in git, see below)
```

The notebooks, in order:

| Notebook | What it does |
|---|---|
| `01_mlp_mnist.py` | LeNet-300-100 on MNIST |
| `02_cnn_cifar10.py` | Conv-4 on CIFAR-10, per-layer pruning |
| `02b_cnn_global.py` | The same, with global pruning |
| `03_resnet18_cifar10.py` | 20-layer CIFAR ResNet (the LTH paper calls it ResNet-18), rewinding to epoch 1 |
| `04_vit_cifar10.py` | Small ViT, 20 epochs, rewinding to epoch 2 |
| `05_vit_longer.py` | The same ViT, 100 epochs and stronger augmentation. This is the main ViT experiment |
| `06_comparison.py` | Reads `results/*.json` and compares everything |
| `07_inspection.py` | Looks inside the saved checkpoints: which weights survive per layer, masks, per-class accuracy, attention |
| `08_demo.py` | The interactive demo |

`lth.py` is the pruning code: masks, magnitude pruning, the iterative loop with optional rewinding, the random control, and saving checkpoints. Each notebook defines its own model and training loop.

A note on `rendered/`: `01_mlp_mnist` and `02_cnn_cifar10` show my first run. The JSON files, the checkpoints and the report come from a rerun that also saved checkpoints, so a few numbers differ (the per-layer CNN at 10.7% got 49.7% the first time and 53.0% the second; the report mentions both). There's no HTML for `05_vit_longer` because exporting it would redo the 5-hour training. Its results are in `results/vit_longer*.json` and show up in `06_comparison` and `07_inspection`.

## Running it

```bash
uv sync
uv run marimo edit notebooks/01_mlp_mnist.py     # any notebook
```

Each training notebook writes its curves to `results/` when it's done, so run 01 to 05 before the comparison. Exporting a notebook to HTML re-runs every cell:

```bash
cd notebooks
uv run marimo export html 06_comparison.py -o ../rendered/06_comparison.html
INSPECT_DIR=../checkpoints/05_vit_longer INSPECT_PCT=10 uv run marimo export html 07_inspection.py -o ../rendered/07_inspection.html
```

### The demo

```bash
cd notebooks && uv run marimo run 08_demo.py
```

You pick a model (CNN per-layer, CNN global or ViT), a pruning level, and a CIFAR-10 test image, or upload your own. It shows what the dense network, the ticket, the rewound ticket (ViT only) and the random control each predict. There's a filter that jumps to images where the ticket gets it right and the control gets it wrong. For the ViT it also shows the attention, and at the bottom the accuracy curves for that experiment. It runs on CPU using the checkpoints in git; the first run downloads CIFAR-10 into `notebooks/data`. `rendered/08_demo.html` is just a snapshot of one setting, so the controls don't do anything there.

### Report, figures and slides

```bash
uv run python report/make_figures.py          # figures and tables for the report
cd report && latexmk -pdf informe.tex          # builds report/informe.pdf
python3 presentation/build.py                  # builds presentation/slides.html
uv run python report/make_figures.py --dark    # architecture diagrams for the slides
```

## Checkpoints

Every pruning round saves `checkpoints/<run>/<tag>_rNN_<pct>pct.pt`, where `tag` is `ticket`, `late` (the rewinding ticket) or `random`. Each file has the weights, the masks, the accuracy and the fraction of weights left. `base.pt` holds the dense, initial and rewind weights. Load them with `lth.load_checkpoint(path)` and list them with `lth.list_checkpoints(dir)`.

The full set is about 1.1 GB, so I only put a ~235 MB subset in git: every `base.pt`, plus the most useful levels (MLP 41%, 10.7% and 3.5%; CNN per-layer 32.8% and 10.7%; CNN global 10.7%; ViT 11.6% and 4.9%). That's enough for the demo. The rest is git-ignored and comes back if you rerun the notebooks. Only runs 01, 02, 02b and 05 have checkpoints (03 and 04 were run before I added the saving code). A few report figures that need every round, like the ViT curves and the survival by block, can't be regenerated from the subset.

## How I used AI

I wrote the four models, the training and evaluation loops and `lth.py` myself. Then I used Claude Code to wire `lth.py` into the other notebooks, run the experiments, add checkpoint saving, make figures and tables, and draft text. Section 5.2 of the report covers what worked, what it got wrong, and how I checked it. Every number in the report comes from `results/*.json` or the checkpoints.

## References

1. Frankle & Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR 2019. https://arxiv.org/abs/1803.03635
2. Frankle, Dziugaite, Roy & Carbin, *Stabilizing the Lottery Ticket Hypothesis*, 2019. https://arxiv.org/abs/1903.01611
3. Renda, Frankle & Carbin, *Comparing Rewinding and Fine-tuning in Neural Network Pruning*, ICLR 2020. https://arxiv.org/abs/2003.02389
4. Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021. https://arxiv.org/abs/2012.12877
5. Shen et al., *Data Level Lottery Ticket Hypothesis for Vision Transformers*, IJCAI 2023. https://arxiv.org/abs/2211.01484
6. Jain & Wallace, *Attention is not Explanation*, NAACL 2019. https://arxiv.org/abs/1902.10186

Where my setup differs from these papers (epochs per round, pruning rates, the ViT learning rate and warmup) is listed in Appendix A of the report.
