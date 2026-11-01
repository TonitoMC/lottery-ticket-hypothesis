"""Genera TODAS las figuras y tablas del informe a partir de results/*.json y checkpoints/.

    cd <raíz del repo>
    uv run python report/make_figures.py          # todo
    uv run python report/make_figures.py --only curves delta   # solo algunas

Salida (report/figures/): un .pdf (para LaTeX) y un .png por figura, tables.tex (tablas de números
para \\input) y index.html (galería con todas las figuras y cómo leerlas).

Las figuras que usan checkpoints del ViT (heat, cls, attn, grad, entropy) leen checkpoints/05_vit_longer
y results/inspection_05_vit_longer.json (este último lo genera notebooks/06_inspection.py).
El código de las figuras fue generado con ayuda de IA; los datos son los de las corridas.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "notebooks"))
import lth  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight"})
COL = {"ticket": "#1f77b4", "ticket_late": "#2ca02c", "random": "#ff7f0e", "dense": "#7f7f7f"}
NAME = {"ticket": "Ticket (época 0)", "ticket_late": "Ticket (rewinding)", "random": "Reinit. aleatorio"}
RUNS = {
    "mlp_mnist": "MLP / MNIST",
    "cnn_cifar10": "CNN poda por capa / CIFAR-10",
    "cnn_global_cifar10": "CNN poda global / CIFAR-10",
    "resnet18_cifar10": "ResNet-20 / CIFAR-10",
    "vit_cifar10": "ViT (20 ép.) / CIFAR-10",
    "vit_longer": "ViT largo (100 ép.) / CIFAR-10",
}
VIT_DIR = ROOT / "checkpoints" / "05_vit_longer"
TARGET = 11.6  # % de pesos al que se comparan los checkpoints del ViT

REGISTRY = []  # (id, título, qué es / cómo leerla)


def figure(fid, title, reading):
    """Decorador: registra la figura y guarda PDF + PNG."""
    def deco(fn):
        fn.meta = (fid, title, reading)
        REGISTRY.append(fn)
        return fn
    return deco


def save(fig, fid):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fid}.pdf")
    fig.savefig(OUT / f"{fid}.png", dpi=170)
    plt.close(fig)


def load_runs():
    runs = {}
    for key, name in RUNS.items():
        p = ROOT / "results" / f"{key}.json"
        if p.exists():
            runs[name] = json.loads(p.read_text())
    return runs


RUNS_DATA = load_runs()


# ---------------------------------------------------------------- arquitecturas
def _row(ax, y, items, title, colors=None):
    ax.text(0.0, y + 0.62, title, fontsize=9.5, fontweight="bold", va="bottom")
    x = 0.0
    for i, (txt, w) in enumerate(items):
        c = (colors or {}).get(i, "#dbe9f6")
        ax.add_patch(FancyBboxPatch((x, y), w, 0.55, boxstyle="round,pad=0.02", fc=c, ec="#33506b", lw=0.8))
        ax.text(x + w / 2, y + 0.275, txt, ha="center", va="center", fontsize=6.8)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + w + 0.18, y + 0.275), xytext=(x + w + 0.02, y + 0.275), arrowprops=dict(arrowstyle="->", color="#33506b", lw=0.8))
        x += w + 0.2


@figure("arch", "Arquitecturas e IMP",
        "Las cuatro redes con la forma de los tensores y los pesos podables por capa (verde = capas que se podan). Abajo, el ciclo de IMP: entrenar, podar el menor |w|, regresar los sobrevivientes al punto de referencia y volver a entrenar.")
def fig_arch():
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(-0.2, 15.2)
    ax.set_ylim(-0.2, 10.4)
    ax.axis("off")
    ax.grid(False)
    P, N, A, V = "#cfe8cf", "#eeeeee", "#fbe3c8", "#e3d8f0"
    _row(ax, 9.0, [("Imagen\n(784)", 1.0), ("Linear\n784→300\n235k", 1.2), ("ReLU", 0.7), ("Linear\n300→100\n30k", 1.2), ("ReLU", 0.7), ("Linear\n100→10\n1k", 1.1), ("10 clases", 1.0)],
         "MLP (LeNet-300-100) · MNIST · 266k pesos podables", {1: P, 3: P, 5: P, 2: A, 4: A})
    _row(ax, 7.2, [("Imagen\n(3,32,32)", 1.1), ("Conv 3→64\n1.7k", 1.0), ("Conv 64→64\n37k", 1.0), ("MaxPool\n(64,16,16)", 0.95), ("Conv 64→128\n74k", 1.05), ("Conv 128→128\n147k", 1.1), ("MaxPool\n(128,8,8)", 0.95), ("Linear\n8192→256\n2.1M", 1.05), ("Linear\n256→256\n66k", 1.0), ("Linear\n256→10", 0.85)],
         "CNN (Conv-4) · CIFAR-10 · ~2.4M pesos podables (86% en la primera Linear)", {1: P, 2: P, 4: P, 5: P, 7: P, 8: P, 9: P, 3: N, 6: N})
    _row(ax, 5.4, [("Imagen\n(3,32,32)", 1.1), ("Conv 3→16\n+BN+ReLU", 1.1), ("3 bloques\n16 canales\n(32×32)", 1.25), ("3 bloques\n32 canales\n(16×16)", 1.2), ("3 bloques\n64 canales\n(8×8)", 1.2), ("AvgPool\n+ Linear 64→10\n(sin poda)", 1.1), ("10 clases", 0.9)],
         "ResNet de 20 capas · CIFAR-10 · ~270k pesos podables (solo convoluciones; la capa final no se poda)", {1: P, 2: P, 3: P, 4: P})
    ax.text(9.6, 5.75, "Bloque residual: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ x) → ReLU\n(con cambio de canales, atajo Conv 1×1)", fontsize=7.2, va="center", color="#444")
    _row(ax, 3.6, [("Imagen\n(3,32,32)", 1.1), ("Parches 4×4\nConv 3→192\n(64 tokens)", 1.3), ("+ [CLS]\n+ posición\n(65,192)", 1.1), ("× 6 bloques\n(ver abajo)", 1.3), ("LayerNorm", 0.9), ("[CLS] →\nLinear\n192→10", 1.0), ("10 clases", 0.9)],
         "ViT pequeño · CIFAR-10 · ~1.8M pesos podables (6 bloques, dim 192, 3 cabezas)", {1: P, 5: P, 3: V})
    _row(ax, 1.8, [("x", 0.5), ("LayerNorm", 0.9), ("qkv\n192→576", 0.9), ("Atención\n3 cabezas", 1.0), ("proj\n192→192", 0.9), ("+ x", 0.55), ("LayerNorm", 0.9), ("mlp.0\n192→384", 0.95), ("GELU", 0.6), ("mlp.2\n384→192", 0.95), ("+ x", 0.55)],
         "Un bloque del ViT (se repite 6 veces); las Linear y las LayerNorm entran a la poda global", {2: P, 4: P, 7: P, 9: P, 3: V, 1: N, 6: N})
    steps = ["1. Entrenar\n(con la máscara)", "2. Podar 20%\nde los pesos vivos\n(menor |w|)", "3. Regresar los\nsobrevivientes a θ₀\no a θ_k (rewind)", "4. Volver a entrenar\ny medir accuracy"]
    x = 0.2
    ax.text(0.0, 1.2, "Poda por magnitud iterativa (IMP)", fontsize=9.5, fontweight="bold")
    for i, t in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.1), 2.7, 0.9, boxstyle="round,pad=0.03", fc="#fff2cc", ec="#7f6000", lw=0.8))
        ax.text(x + 1.35, 0.55, t, ha="center", va="center", fontsize=7.2)
        if i < 3:
            ax.annotate("", xy=(x + 3.15, 0.55), xytext=(x + 2.76, 0.55), arrowprops=dict(arrowstyle="->", lw=1))
        x += 3.2
    ax.text(13.1, 0.55, "se repite\n8-15 rondas", ha="left", va="center", fontsize=8, fontweight="bold")
    ax.text(15.1, 10.3, "Verde: se poda · Naranja: activación · Gris: sin poda · Morado: atención", ha="right", fontsize=7, color="#444")
    save(fig, "arch")


# ---------------------------------------------------------------- curvas
@figure("curves", "Accuracy vs. pesos restantes (todas las corridas)",
        "Cada panel es una arquitectura. Eje x: % de pesos que quedan (log, invertido: la red se hace más pequeña hacia la derecha). Si la curva azul se mantiene arriba y la naranja cae, el ticket funciona; si ambas caen igual, solo importaba la estructura.")
def fig_curves():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
    for ax, (name, run) in zip(axes.flat, RUNS_DATA.items()):
        for key in ("ticket", "ticket_late", "random"):
            if key in run:
                res = run[key]
                ax.plot([r["weights_left"] * 100 for r in res], [r["acc"] for r in res], marker="o", ms=3.5, color=COL[key], ls="--" if key == "random" else "-", label=NAME[key])
        ax.axhline(run["ticket"][0]["acc"], color=COL["dense"], lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Pesos restantes (%)")
    for ax in list(axes.flat)[len(RUNS_DATA):]:
        ax.axis("off")
    axes[0][0].set_ylabel("Accuracy de prueba")
    axes[1][0].set_ylabel("Accuracy de prueba")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL[k], marker="o", ms=3.5, ls="--" if k == "random" else "-", label=NAME[k]) for k in ("ticket", "ticket_late", "random")]
    handles.append(Line2D([], [], color=COL["dense"], ls=":", label="Red densa"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "curves")


@figure("delta", "Accuracy relativa a la red densa",
        "A cada curva se le resta la accuracy de su red densa, para comparar arquitecturas con bases distintas. 0 = la subred iguala a la completa; arriba de 0 = la supera. Línea continua: ticket; punteada: control aleatorio.")
def fig_delta():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (name, run) in enumerate(RUNS_DATA.items()):
        dense = run["ticket"][0]["acc"]
        c = f"C{i}"
        ax.plot([r["weights_left"] * 100 for r in run["ticket"]], [(r["acc"] - dense) * 100 for r in run["ticket"]], marker="o", ms=3, color=c, label=name)
        if "random" in run:
            ax.plot([r["weights_left"] * 100 for r in run["random"]], [(r["acc"] - dense) * 100 for r in run["random"]], ls="--", color=c, alpha=0.6)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Pesos restantes (%)")
    ax.set_ylabel("Δ accuracy vs. red densa (puntos)")
    ax.legend(fontsize=7)
    save(fig, "delta")


@figure("vit_dense", "ViT denso de 100 épocas: entrenamiento vs. prueba",
        "Accuracy de entrenamiento (sin aumento de datos) y de prueba por época. Una brecha pequeña con la prueba todavía subiendo indica un modelo no fuertemente sobreajustado, es decir, un baseline razonable para comparar el ticket.")
def fig_vit_dense():
    p = ROOT / "results" / "vit_longer_dense.json"
    h = json.loads(p.read_text())["history"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot([e["epoch"] for e in h], [e["train_acc"] for e in h], label="Entrenamiento (sin aumento)")
    ax.plot([e["epoch"] for e in h], [e["test_acc"] for e in h], label="Prueba")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.legend()
    save(fig, "vit_dense")


# ---------------------------------------------------------------- CNN: por capa vs global
def cnn_layer_info():
    dirs = {"Por capa (02)": ROOT / "checkpoints" / "02_cnn", "Global (02b)": ROOT / "checkpoints" / "02b_cnn_global"}
    info = {}
    for lab, d in dirs.items():
        ck = torch.load(next(d.glob("ticket_r10_*.pt")), map_location="cpu")
        info[lab] = {}
        for n, m in ck["masks"].items():
            key = n.removesuffix(".weight")
            row = {"keep": m.float().mean().item(), "n": int(m.sum()), "total": m.numel()}
            if m.dim() == 4:
                row["dead"], row["of"] = int((m.flatten(1).sum(1) == 0).sum()), m.shape[0]
            elif key == "linear.5":
                row["dead"], row["of"] = int((m.sum(0) == 0).sum()), m.shape[1]
            info[lab][key] = row
    return info


CNN_LAYERS = ["conv.0", "conv.2", "conv.5", "conv.7", "linear.1", "linear.3", "linear.5"]
CNN_NICE = ["conv1 (3→64)", "conv2 (64→64)", "conv3 (64→128)", "conv4 (128→128)", "fc1 (8192→256)", "fc2 (256→256)", "fc3 (256→10)"]


@figure("cnn_layers", "CNN: qué sobrevive por capa, poda por capa vs. global",
        "Fracción de pesos vivos de cada capa en el ticket con 10.7% de los pesos totales (línea punteada). Con poda por capa todas las barras quedan en ~10.7%, incluso las capas diminutas (conv1 pasa de 1,728 a 188 pesos). Con poda global las capas pequeñas casi no se podan y fc1 absorbe la mayor parte de la poda.")
def fig_cnn_layers():
    info = cnn_layer_info()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.38
    for i, lab in enumerate(info):
        ax.bar([x + i * w for x in range(7)], [info[lab][k]["keep"] * 100 for k in CNN_LAYERS], w, label=lab)
    ax.axhline(10.7, color="gray", ls=":", lw=1)
    ax.set_xticks([x + w / 2 for x in range(7)])
    ax.set_xticklabels(CNN_NICE, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pesos vivos de la capa (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "cnn_layers")


# ---------------------------------------------------------------- ViT (checkpoints)
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

    def all_attention(self, x):
        x, out = self.embed(x), []
        for blk in self.blocks:
            out.append(blk.attention(x))
            x = blk(x)
        return out


def vit(state):
    m = ViT()
    m.load_state_dict(state)
    return m.eval()


def layer_type(n):
    return re.sub(r"^blocks\.\d+\.", "", n).removesuffix(".weight")


_cache = {}


def vit_ctx():
    """Checkpoints y datos del ViT largo, cargados una sola vez."""
    if _cache:
        return _cache
    from torchvision import datasets
    base = torch.load(VIT_DIR / "base.pt", map_location="cpu")
    curves = {}
    for tag, r, pct, acc, path in lth.list_checkpoints(str(VIT_DIR)):
        curves.setdefault(tag, []).append((r, pct, acc, path))
    for t in curves:
        curves[t].sort()
    test = datasets.CIFAR10(root=str(ROOT / "notebooks" / "data"), train=False, download=True)
    mean, std = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1), torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
    _cache.update(base=base, curves=curves, test=test, mean=mean, std=std,
                  at=lambda tag: min(curves[tag], key=lambda c: abs(c[1] - TARGET)),
                  load=lambda p: torch.load(p, map_location="cpu"))
    return _cache


def img_tensor(test, idx):
    return torch.tensor(np.array(test[idx][0])).permute(2, 0, 1).float() / 255


@figure("heat", "ViT largo: fracción viva por tipo de capa y ronda",
        "Para cada tipo de capa (promedio ponderado sobre los 6 bloques) y cada ronda, la fracción de pesos que sobrevive. Con poda global las capas no pierden parejo: las filas se separan. Una capa que se oscurece rápido es una capa cuyos pesos son pequeños en comparación con el resto de la red.")
def fig_heat():
    c = vit_ctx()
    rows = c["curves"]["ticket"]
    m0 = c["load"](rows[0][3])["masks"]
    types = []
    for n, m in m0.items():
        if m.dim() >= 2 and layer_type(n) not in types:
            types.append(layer_type(n))
    grid = np.zeros((len(types), len(rows)))
    for j, (_, _, _, p) in enumerate(rows):
        ms = c["load"](p)["masks"]
        for i, t in enumerate(types):
            alive = sum(m.sum().item() for n, m in ms.items() if layer_type(n) == t and m.dim() >= 2)
            total = sum(m.numel() for n, m in ms.items() if layer_type(n) == t and m.dim() >= 2)
            grid[i, j] = alive / total
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.grid(False)
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r[1]:.0f}%" if r[1] >= 10 else f"{r[1]:.1f}%" for r in rows], fontsize=8)
    ax.set_xlabel("Pesos que quedan en total")
    for i in range(len(types)):
        for j in range(len(rows)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if grid[i, j] < 0.6 else "k")
    fig.colorbar(im, label="Fracción viva")
    save(fig, "heat")
    c["types"] = types


@figure("cls", "ViT largo: accuracy por clase respecto al denso",
        "Diferencia (en puntos) entre cada modelo podado y el denso, por clase, con 11.6% de los pesos. La banda gris (±2.5 puntos) es el ruido estadístico de 1,000 imágenes por clase: lo que cae dentro no se interpreta.")
def fig_cls():
    c = vit_ctx()
    ins = json.loads((ROOT / "results" / "inspection_05_vit_longer.json").read_text())
    key = lambda tag: f"{tag}_r{c['at'](tag)[0]}"
    dense = np.array(ins["dense_r0"]["per_class"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    w = 0.26
    for i, (tag, ck) in enumerate((("ticket", "ticket"), ("late", "ticket_late"), ("random", "random"))):
        ax.bar(np.arange(10) + i * w, (np.array(ins[key(tag)]["per_class"]) - dense) * 100, w, color=COL[ck], label=f"{NAME[ck]} ({ins[key(tag)]['acc'] * 100:.1f}%)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axhspan(-2.5, 2.5, color="gray", alpha=0.15)
    ax.set_xticks(np.arange(10) + w)
    ax.set_xticklabels(c["test"].classes, rotation=40, fontsize=8)
    ax.set_ylabel("Puntos vs. denso (86.0%)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "cls")


@figure("attn", "ViT largo: mapas de atención del [CLS]",
        "Atención del token [CLS] hacia los 64 parches (8×8) en el último bloque, promediada sobre las 3 cabezas, para 5 imágenes de prueba (columnas). Cada fila es un modelo con 11.6% de los pesos; 'aleatorio' tiene la misma máscara que el ticket con otra inicialización. Más brillante = más atención. Es una pista de dónde mira el modelo, no una explicación de su decisión.")
def fig_attn():
    c = vit_ctx()
    t = torch.tensor(c["test"].targets)
    pick = [(t == k).nonzero()[0].item() for k in (0, 1, 3, 5, 8)]
    imgs = torch.stack([img_tensor(c["test"], i) for i in pick])
    x = (imgs - c["mean"]) / c["std"]
    models = [("denso", vit(c["base"]["dense"]))] + [(n, vit(c["load"](c["at"](t_)[3])["model"])) for n, t_ in (("ticket", "ticket"), ("rewinding", "late"), ("aleatorio", "random"))]
    fig, axes = plt.subplots(len(models) + 1, len(pick), figsize=(5.8, 5.0))
    for j in range(len(pick)):
        axes[0, j].imshow(imgs[j].permute(1, 2, 0))
    axes[0, 0].set_ylabel("imagen", fontsize=8)
    for i, (n, m) in enumerate(models, start=1):
        with torch.no_grad():
            a = m.all_attention(x)[-1][:, :, 0, 1:].mean(1)
        for j in range(len(pick)):
            axes[i, j].imshow(a[j].reshape(8, 8), cmap="inferno")
        axes[i, 0].set_ylabel(n, fontsize=8)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.tight_layout()
    save(fig, "attn")


@figure("grad", "ViT largo: norma del gradiente en el paso 0",
        "Norma del gradiente de la pérdida sobre los pesos vivos, por tipo de capa, antes de entrenar (eje logarítmico). Compara la misma capa entre modelos, no capas distintas, porque la norma crece con el número de pesos.")
def fig_grad():
    c = vit_ctx()
    types = c.get("types")
    if types is None:
        fig_heat()
        types = c["types"]
    idx = range(512)
    sample = torch.stack([img_tensor(c["test"], i) for i in idx])
    sample = (sample - c["mean"]) / c["std"]
    sy = torch.tensor(c["test"].targets[:512])

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

    ck = {t: c["load"](c["at"](t)[3]) for t in ("ticket", "late", "random")}
    g0 = {"ticket": grads(masked(c["base"]["init"], ck["ticket"]["masks"]), ck["ticket"]["masks"]),
          "ticket_late": grads(masked(c["base"]["rewind"], ck["late"]["masks"]), ck["late"]["masks"]),
          "random": grads(vit(ck["random"]["init_model"]), ck["random"]["masks"])}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.8 / 3
    for i, (t, d) in enumerate(g0.items()):
        ax.bar(np.arange(len(types)) + i * w, [d[k] for k in types], w, label=NAME[t], color=COL[t])
    ax.set_xticks(np.arange(len(types)) + w)
    ax.set_xticklabels(types, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Norma del gradiente (paso 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "grad")


@figure("entropy", "ViT largo: entropía de la atención vs. pesos restantes",
        "Entropía de la fila del [CLS] (65 llaves), promediada sobre 256 imágenes, cabezas y bloques. Máxima (ln 65 ≈ 4.17) = atención uniforme; cercana a 0 = una sola llave. Si cambia al podar, las cabezas están cambiando de comportamiento; no dice si la atención es correcta.")
def fig_entropy():
    c = vit_ctx()
    sample = torch.stack([img_tensor(c["test"], i) for i in range(256)])
    sample = (sample - c["mean"]) / c["std"]

    def ent(model):
        with torch.no_grad():
            att = model.all_attention(sample)
        return float(np.mean([-(a * (a + 1e-12).log()).sum(-1)[:, :, 0].mean().item() for a in att]))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    dense_e = ent(vit(c["base"]["dense"]))
    for tag, key in (("ticket", "ticket"), ("late", "ticket_late"), ("random", "random")):
        rows = c["curves"][tag]
        ax.plot([r[1] for r in rows], [ent(vit(c["load"](r[3])["model"])) for r in rows], marker="o", ms=3.5, color=COL[key], label=NAME[key], ls="--" if tag == "random" else "-")
    ax.axhline(dense_e, color=COL["dense"], ls=":", label="Denso")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Pesos restantes (%)")
    ax.set_ylabel("Entropía de la atención del [CLS] (nats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "entropy")


@figure("magnitudes", "Tamaño de los pesos por capa y qué sobrevive a la poda global",
        "Panel izquierdo: mediana de |w| por capa en la CNN densa entrenada (eje log; el tamaño de la capa está sobre cada barra). Panel central: mediana de |w| contra el porcentaje de pesos vivos de esa capa en el ticket global de la CNN (10.7% del total): las capas con pesos grandes (pocas entradas por neurona) sobreviven más. Panel derecho: lo mismo para el ViT largo (11.6% del total); la relación también es positiva (head y patch_embed tienen los pesos más grandes y sobreviven más), pero es menos limpia porque las magnitudes cambian durante las rondas de IMP y aquí se usan las del modelo denso.")
def fig_magnitudes():
    # CNN: pesos densos entrenados (02b) y máscara del ticket global con 10.7%
    base = torch.load(ROOT / "checkpoints" / "02b_cnn_global" / "base.pt", map_location="cpu")
    ck = torch.load(next((ROOT / "checkpoints" / "02b_cnn_global").glob("ticket_r10_*.pt")), map_location="cpu")
    med = {k: float(base["dense"][k + ".weight"].abs().median()) for k in CNN_LAYERS}
    size = {k: base["dense"][k + ".weight"].numel() for k in CNN_LAYERS}
    keep = {k: ck["masks"][k + ".weight"].float().mean().item() * 100 for k in CNN_LAYERS}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ax = axes[0]
    ax.bar(range(7), [med[k] for k in CNN_LAYERS], color="#1f77b4")
    for i, k in enumerate(CNN_LAYERS):
        ax.text(i, med[k] * 1.12, f"{size[k]:,}", ha="center", fontsize=6.5, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=0.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(CNN_NICE, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mediana de |w| (red densa)")
    ax.set_title("CNN: tamaño de los pesos", fontsize=9)
    ax = axes[1]
    for k, nm in zip(CNN_LAYERS, CNN_NICE):
        ax.scatter(med[k], keep[k], s=30, color="#ff7f0e")
        ax.annotate(nm.split(" ")[0], (med[k], keep[k]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mediana de |w| (red densa)")
    ax.set_ylabel("% vivo de la capa (ticket global, 10.7%)")
    ax.set_title("CNN: pesos más grandes, sobreviven más", fontsize=9)
    c = vit_ctx()
    vb = c["base"]["dense"]
    vmask = c["load"](c["at"]("ticket")[3])["masks"]
    types, vmed, vkeep = [], {}, {}
    for n, w in vb.items():
        if n.endswith("weight") and w.dim() >= 2:
            t = layer_type(n)
            vmed.setdefault(t, []).append(w.abs().flatten())
            vkeep.setdefault(t, []).append(vmask[n].flatten().float())
    ax = axes[2]
    for t in vmed:
        m = float(torch.cat(vmed[t]).median())
        kp = float(torch.cat(vkeep[t]).mean()) * 100
        ax.scatter(m, kp, s=30, color="#2ca02c")
        ax.annotate(t, (m, kp), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mediana de |w| (ViT denso)")
    ax.set_ylabel("% vivo (ticket, 11.6%)")
    ax.set_title("ViT largo", fontsize=9)
    fig.tight_layout()
    save(fig, "magnitudes")


# ---------------------------------------------------------------- tablas (LaTeX)
def make_tables():
    OUT.mkdir(parents=True, exist_ok=True)

    def smallest(res, dense):
        ok = [r["weights_left"] for r in res if r["acc"] >= dense - 0.01]
        return f"{min(ok) * 100:.1f}\\%" if ok else "ninguna"

    def mean_gap(a, b):
        n = min(len(a), len(b)) - 1
        return sum(x["acc"] - y["acc"] for x, y in zip(a[1:], b[1:])) / n * 100 if n > 0 else float("nan")

    rows = []
    for name, run in RUNS_DATA.items():
        dense = run["ticket"][0]["acc"]
        rnd = run.get("random", [])
        adv = f"{mean_gap(run['ticket'], rnd):+.1f}" if len(rnd) > 1 else "--"
        late = f"{mean_gap(run['ticket_late'], run['ticket']):+.1f}" if len(run.get("ticket_late", [])) > 1 else "--"
        rows.append(f"{name} & {dense * 100:.1f} & {smallest(run['ticket'], dense)} & {smallest(rnd, dense) if rnd else '--'} & {adv} & {late} \\\\")
    summary = ("\\begin{tabular}{@{}lrrrrr@{}}\\toprule\nArquitectura & Densa (\\%) & M\\'in. pesos ticket & M\\'in. pesos aleat. & Ventaja ticket (pts) & Ganancia rewind (pts) \\\\\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    (OUT / "table_summary.tex").write_text(summary)

    # Tabla del ViT largo en puntos seleccionados + CNN a 10.7%
    vl = RUNS_DATA.get(RUNS["vit_longer"])
    if vl:
        lines = []
        for i, r in enumerate(vl["ticket"]):
            late = vl["ticket_late"][i]["acc"] * 100 if i < len(vl.get("ticket_late", [])) else float("nan")
            rnd = vl["random"][i]["acc"] * 100 if i < len(vl["random"]) else float("nan")
            lines.append(f"{r['weights_left'] * 100:.1f} & {r['acc'] * 100:.1f} & {late:.1f} & {rnd:.1f} \\\\")
        (OUT / "table_vitlong.tex").write_text("\\begin{tabular}{@{}rrrr@{}}\\toprule\n\\% de pesos & Ticket $\\theta_0$ & Ticket rewind (ép. 5) & Aleatorio \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")

    info = cnn_layer_info()
    lines = []
    for k, nm in zip(CNN_LAYERS, CNN_NICE):
        cells = []
        for lab in info:
            r = info[lab][k]
            cells += [f"{r['keep'] * 100:.1f}", f"{r['n']:,}".replace(",", "\\,"), f"{r['dead']}/{r['of']}" if "dead" in r else "--"]
        lines.append(f"{nm.replace(chr(8594), '$\\to$')} & " + " & ".join(cells) + " \\\\")
    (OUT / "table_cnnmasks.tex").write_text("\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n & \\multicolumn{3}{c}{Por capa (02)} & \\multicolumn{3}{c}{Global (02b)} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nCapa & \\% vivo & Pesos & Vac\\'ios & \\% vivo & Pesos & Vac\\'ios \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    print("tablas escritas")


# ---------------------------------------------------------------- galería HTML
def make_index(done):
    cards = []
    for fid, title, reading in done:
        cards.append(f'<section><h2>{html.escape(title)}</h2><img src="{fid}.png" alt="{html.escape(title)}"><p><b>Cómo leerla.</b> {html.escape(reading)}</p><p class="f">{fid}.pdf</p></section>')
    page = ("<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Figuras del informe</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;line-height:1.5;background:#fff;color:#111}"
            "section{margin:2rem 0;padding-bottom:1.5rem;border-bottom:1px solid #ddd}img{max-width:100%}.f{color:#777;font-size:.85em}"
            "@media (prefers-color-scheme:dark){body{background:#15171c;color:#e6e6e6}section{border-color:#333}img{background:#fff}}</style></head><body>"
            "<h1>Figuras del informe</h1><p>Generadas por <code>report/make_figures.py</code> a partir de <code>results/</code> y <code>checkpoints/</code>. "
            "Cada una trae una explicación de cómo leerla.</p>" + "".join(cards) + "</body></html>")
    (OUT / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="ids de figuras (por defecto todas)")
    args = ap.parse_args()
    done = []
    for fn in REGISTRY:
        fid, title, reading = fn.meta
        if args.only and fid not in args.only:
            continue
        print("figura:", fid, flush=True)
        fn()
        done.append(fn.meta)
    if not args.only:
        make_tables()
        make_index(done)
        # diagramas de la CNN y del ViT (comparten código con la presentación; modo claro para el informe)
        import subprocess
        for script in ("make_cnn_figs.py", "make_vit_figs.py"):
            subprocess.run([sys.executable, str(ROOT / "presentation" / script), "--light"], check=True)
    print("listo:", OUT)


if __name__ == "__main__":
    main()
