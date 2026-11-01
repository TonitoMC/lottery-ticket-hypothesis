"""Figuras de la diapositiva 9 (inspección del ViT largo). Lee checkpoints/05_vit_longer y results/inspection_05_vit_longer.json.

    cd notebooks && uv run python ../presentation/make_figs.py

El código de las figuras fue generado con IA; las mismas vistas (con más detalle) están en notebooks/06_inspection.py.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "notebooks"))
import lth  # noqa: E402

RUN = root / "checkpoints" / "05_vit_longer"
OUT = Path(__file__).parent / "img"
TARGET = 11.6
BG, INK, MUTED = "#1c2140", "#f2f3f8", "#a3acc9"
C = {"dense": "#a3acc9", "ticket": "#6ea8ff", "late": "#3ddc97", "random": "#ffb454"}
NAMES = {"dense": "denso", "ticket": "ticket", "late": "rewind", "random": "aleatorio"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 9})


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.heads = heads
        self.norm1, self.qkv, self.proj, self.norm2 = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def attention(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        return torch.softmax(q @ k.transpose(-2, -1) / (D // self.heads) ** 0.5, dim=-1)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, dim=192, depth=6, heads=3, mlp_dim=384, patch=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, (32 // patch) ** 2 + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, 10)

    def embed(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.embed(x)))[:, 0])

    def last_attention(self, x):
        x = self.embed(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        return self.blocks[-1].attention(x)


def vit(state):
    m = ViT()
    m.load_state_dict(state)
    return m.eval()


def layer_type(n):
    return re.sub(r"^blocks\.\d+\.", "", n).removesuffix(".weight")


base = torch.load(RUN / "base.pt", map_location="cpu")
curves = {}
for tag, r, pct, acc, path in lth.list_checkpoints(str(RUN)):
    curves.setdefault(tag, []).append((r, pct, acc, path))
for t in curves:
    curves[t].sort()
at = lambda tag: min(curves[tag], key=lambda c: abs(c[1] - TARGET))
load = lambda p: torch.load(p, map_location="cpu")

# 1. Fracción viva por tipo de capa y ronda (promedio ponderado sobre los bloques)
rows = curves["ticket"]
masks0 = load(rows[0][3])["masks"]
types = []
for n, m in masks0.items():
    if m.dim() >= 2 and layer_type(n) not in types:
        types.append(layer_type(n))
grid = np.zeros((len(types), len(rows)))
for j, (_, _, _, p) in enumerate(rows):
    ms = load(p)["masks"]
    for i, t in enumerate(types):
        alive = sum(m.sum().item() for n, m in ms.items() if layer_type(n) == t and m.dim() >= 2)
        total = sum(m.numel() for n, m in ms.items() if layer_type(n) == t and m.dim() >= 2)
        grid[i, j] = alive / total
fig, ax = plt.subplots(figsize=(9, 2.7))
im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
ax.set_yticks(range(len(types))); ax.set_yticklabels(types)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{c[1]:.0f}%" if c[1] >= 10 else f"{c[1]:.1f}%" for c in rows], fontsize=8)
ax.set_xlabel("Pesos que quedan en total")
for i in range(len(types)):
    for j in range(len(rows)):
        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if grid[i, j] < 0.6 else "k")
fig.colorbar(im, label="Fracción viva")
fig.tight_layout(); fig.savefig(OUT / "heat.png", dpi=200); plt.close(fig)
final = dict(zip(types, grid[:, -1]))
print("capa -> fracción viva en la última ronda:", {k: round(v, 3) for k, v in final.items()})

# 2. Atención del CLS en el último bloque
test = datasets.CIFAR10(root=str(root / "notebooks" / "data"), train=False, download=False)
mean, std = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1), torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
targets = torch.tensor(test.targets)
pick = [(targets == c).nonzero()[0].item() for c in (0, 1, 3, 5, 8)]
imgs = torch.stack([torch.tensor(np.array(test[i][0])).permute(2, 0, 1).float() / 255 for i in pick])
x = (imgs - mean) / std
models = [("dense", vit(base["dense"]))] + [(t, vit(load(at(t)[3])["model"])) for t in ("ticket", "late", "random")]
fig, axes = plt.subplots(len(models) + 1, len(pick), figsize=(5.2, 4.4))
for j in range(len(pick)):
    axes[0, j].imshow(imgs[j].permute(1, 2, 0))
axes[0, 0].set_ylabel("imagen", fontsize=8, color=INK)
for i, (tag, m) in enumerate(models, start=1):
    with torch.no_grad():
        a = m.last_attention(x)[:, :, 0, 1:].mean(1)
    for j in range(len(pick)):
        axes[i, j].imshow(a[j].reshape(8, 8), cmap="inferno")
    axes[i, 0].set_ylabel(NAMES[tag], fontsize=8, color=C[tag])
for a in axes.flat:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "attn.png", dpi=200); plt.close(fig)

# 3. Norma del gradiente por tipo de capa, paso 0 (pesos vivos)
sample = torch.stack([torch.tensor(np.array(test[i][0])).permute(2, 0, 1).float() / 255 for i in range(512)])
sample = (sample - mean) / std
sy = torch.tensor(test.targets[:512])


def grads(model, masks):
    model.zero_grad()
    nn.CrossEntropyLoss()(model(sample), sy).backward()
    d = {}
    for n, p in model.named_parameters():
        if n in masks and masks[n].dim() >= 2:
            d.setdefault(layer_type(n), []).append((p.grad * masks[n]).norm().item())
    return {k: float(np.mean(v)) for k, v in d.items()}


def masked(state, masks):
    m = vit(state)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in masks:
                p.mul_(masks[n])
    return m


ck = {t: load(at(t)[3]) for t in ("ticket", "late", "random")}
g0 = {"ticket": grads(masked(base["init"], ck["ticket"]["masks"]), ck["ticket"]["masks"]),
      "late": grads(masked(base["rewind"], ck["late"]["masks"]), ck["late"]["masks"]),
      "random": grads(vit(ck["random"]["init_model"]), ck["random"]["masks"])}
fig, ax = plt.subplots(figsize=(5, 2.8))
w = 0.8 / len(g0)
for i, (t, d) in enumerate(g0.items()):
    ax.bar(np.arange(len(types)) + i * w, [d[k] for k in types], w, label=NAMES[t], color=C[t])
ax.set_xticks(np.arange(len(types)) + 0.27); ax.set_xticklabels(types, fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("Norma del gradiente (paso 0)")
ax.legend(fontsize=8, facecolor=BG, edgecolor=MUTED, labelcolor=INK)
fig.tight_layout(); fig.savefig(OUT / "grad.png", dpi=200); plt.close(fig)
print("gradiente paso 0:", {t: {k: round(v, 3) for k, v in d.items()} for t, d in g0.items()})

# 4. Accuracy por clase, diferencia contra el denso, al porcentaje objetivo
ins = json.loads((root / "results" / "inspection_05_vit_longer.json").read_text())
key = lambda tag: f"{tag}_r{at(tag)[0]}"
dense = np.array(ins["dense_r0"]["per_class"])
classes = test.classes
fig, ax = plt.subplots(figsize=(5, 2.8))
w = 0.26
for i, tag in enumerate(("ticket", "late", "random")):
    ax.bar(np.arange(10) + i * w, (np.array(ins[key(tag)]["per_class"]) - dense) * 100, w, label=f"{NAMES[tag]} ({ins[key(tag)]['acc'] * 100:.1f}%)", color=C[tag])
ax.axhline(0, color=MUTED, lw=0.6)
ax.axhspan(-2.5, 2.5, color=MUTED, alpha=0.15)
ax.set_xticks(np.arange(10) + w); ax.set_xticklabels(classes, rotation=40, fontsize=7.5)
ax.set_ylabel("Puntos vs. denso (86.0%)")
ax.legend(fontsize=8, facecolor=BG, edgecolor=MUTED, labelcolor=INK)
fig.tight_layout(); fig.savefig(OUT / "cls.png", dpi=200); plt.close(fig)
for tag in ("ticket", "late", "random"):
    d = (np.array(ins[key(tag)]["per_class"]) - dense) * 100
    print(tag, "mejor/peor clase vs denso:", classes[d.argmax()], round(d.max(), 1), classes[d.argmin()], round(d.min(), 1))
