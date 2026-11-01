# Lottery Ticket Hypothesis Across Architectures

Lottery Ticket Hypothesis across architectures (MLP, CNN, ResNet-18, ViT): measuring when winning tickets lock in during training, and whether it's driven by architecture or early-training instability.

## Notebooks

| Notebook | Architecture | Dataset |
|---|---|---|
| `01_mlp_mnist.py` | LeNet-300-100 | MNIST |
| `02_cnn_cifar10.py` | Conv-4 | CIFAR-10 |
| `03_resnet18_cifar10.py` | ResNet-18 (+ late rewinding) | CIFAR-10 |
| `04_vit_cifar10.py` | Small ViT (+ late rewinding) | CIFAR-10 |
| `05_comparison.py` | Reads `results/*.json` from 01-04 | - |

`notebooks/lth.py` holds the pruning code (masks, magnitude pruning, IMP loop, random-reinit control). Each notebook defines its own model and training loop.

Sources for the hyperparameters and where we deviate from them: [`REFERENCES.md`](REFERENCES.md).

## How to run

```bash
uv sync
uv run marimo edit notebooks/01_mlp_mnist.py
```

Each notebook saves its curves to `results/` when it finishes; run 01-04 before 05.
