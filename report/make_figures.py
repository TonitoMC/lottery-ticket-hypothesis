"""Genera TODAS las figuras y tablas del informe a partir de results/*.json y checkpoints/.

    cd <raíz del repo>
    uv run python report/make_figures.py          # todo
    uv run python report/make_figures.py --only curves delta   # solo algunas

Salida (report/figures/): un .pdf (para LaTeX) y un .png por figura, tables.tex (tablas de números
para \\input) y index.html (galería con todas las figuras y cómo leerlas).

Las figuras que usan checkpoints del ViT (heat, cls, attn, grad, entropy) leen checkpoints/05_vit_longer
y results/inspection_05_vit_longer.json (este último lo genera notebooks/07_inspection.py).
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


DARK = "--dark" in sys.argv  # versiones para las diapositivas: fondo transparente, colores del tema oscuro (solo los diagramas)
_DARK_MAP = {
    # texto y trazos
    "#000000": "#f2f3f8", "#444444": "#c8cde0", "#555555": "#a3acc9", "#666666": "#a3acc9", "#777777": "#a3acc9", "#888888": "#8f98b8",
    "#1d7a54": "#3ddc97", "#2a9d6f": "#3ddc97", "#d97f00": "#ffb454", "#8e5fd0": "#c9a7ff", "#1f77b4": "#6ea8ff", "#bbbbbb": "#4a5280", "#aaaaaa": "#4a5280",
    # rellenos
    "#d6efdf": "#1b4a3f", "#eeeeee": "#2b3157", "#fde6c8": "#523c1a", "#e6dcf5": "#3a2d63", "#e8f0fb": "#213a6e",
    "#fbd9a5": "#7a5316", "#dce8f7": "#2a4a82", "#c4d8f0": "#34599a",
}


def _dk(c):
    from matplotlib.colors import to_hex, to_rgba
    try:
        r = to_rgba(c)
    except ValueError:
        return c
    if r[3] == 0:
        return c
    return _DARK_MAP.get(to_hex(r[:3]).lower(), c)


def _darken(fig):
    for ax in fig.axes:
        for t in ax.texts:
            t.set_color(_dk(t.get_color()))
            ap = getattr(t, "arrow_patch", None)
            if ap is not None:
                ap.set_edgecolor(_dk(ap.get_edgecolor())); ap.set_facecolor(_dk(ap.get_facecolor()))
        for p in ax.patches:
            fc = p.get_facecolor()
            p.set_facecolor(_dk(fc)); p.set_edgecolor(_dk(p.get_edgecolor()))
        for l in ax.lines:
            l.set_color(_dk(l.get_color()))


def save(fig, fid):
    kw = dict(bbox_inches="tight", pad_inches=0.04) if fid.startswith("arch_") or fid == "imp" else {}
    if DARK:
        if fid.startswith("arch_"):
            d = ROOT / "presentation" / "img"
            _darken(fig)
            fig.savefig(d / f"{fid}.png", dpi=220, transparent=True, **kw)
        plt.close(fig)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fid}.pdf", **kw)
    fig.savefig(OUT / f"{fid}.png", dpi=170, **kw)
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


@figure("imp", "Cómo funciona IMP",
        "Arriba, la máscara de una capa ilustrativa (12×12 pesos aleatorios) a lo largo de las rondas: en cada una se elimina el 20% de los pesos vivos de menor |w| (blanco = podado; más oscuro = mayor |w|). Abajo, el ciclo: entrenar, podar, regresar los sobrevivientes a un valor de referencia (θ₀ = ticket, θ_k = rewinding, pesos nuevos = control) y repetir. Es un esquema: los pesos son aleatorios, no de una corrida real.")
def fig_imp():
    rng = np.random.default_rng(3)
    w = np.abs(rng.normal(size=(12, 12)))
    show = [0, 1, 2, 4, 10]
    fig = plt.figure(figsize=(6.5, 3.0))
    gs = fig.add_gridspec(2, len(show), height_ratios=[1.35, 1], hspace=0.05, wspace=0.12, left=0.02, right=0.98, top=0.93, bottom=0.02)
    for j, r in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        frac = 0.8 ** r
        keep = max(1, int(round(frac * w.size)))
        thr = np.sort(w.ravel())[::-1][keep - 1]  # equivale a podar 20% de los vivos de menor |w| en cada ronda
        ax.imshow(np.where(w >= thr, w, np.nan), cmap="Blues", vmin=0, vmax=2.6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#aaa"); sp.set_linewidth(0.6)
        ax.set_title(f"ronda {r}: {frac * 100:.0f}%" if frac > 0.2 else f"ronda {r}: {frac * 100:.1f}%", fontsize=7.5, pad=3)
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    labels = ["1. Entrenar", "2. Podar 20%\nde menor |w|", "3. Regresar a\nθ₀ o θ_k", "4. Repetir con\nla nueva máscara"]
    xs = [0.1, 2.65, 5.2, 7.75]
    for x, t in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x, 1.15), 2.15, 1.25, boxstyle="round,pad=0.03", fc="#fff2cc", ec="#7f6000", lw=0.8))
        ax.text(x + 1.075, 1.775, t, ha="center", va="center", fontsize=7.5)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 2.62, 1.775), xytext=(x + 2.2, 1.775), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333"))
    ax.plot([8.825, 8.825, 1.175], [1.1, 0.55, 0.55], color="#888", lw=0.8, ls="--")
    ax.annotate("", xy=(1.175, 1.1), xytext=(1.175, 0.55), arrowprops=dict(arrowstyle="->", lw=0.8, color="#888"))
    ax.text(5.0, 0.2, "hasta llegar al % de pesos deseado", ha="center", va="center", fontsize=7, color="#555")
    save(fig, "imp")
    plt.close(fig)


# ---------------------------------------------------------------- diagramas por arquitectura
KIND = {"p": ("#d6efdf", "#2a9d6f"), "n": ("#eeeeee", "#888888"), "a": ("#fde6c8", "#d97f00"), "v": ("#e6dcf5", "#8e5fd0"), "i": ("#e8f0fb", "#1f77b4")}
LEGEND = "verde: capa con pesos que se podan · gris: sin poda · naranja: activación · morado: atención"


def _blk(ax, x, y, w, h, text, kind="p", fs=6.8, bold=False):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)


def _arr(ax, x0, y0, x1, y1, color="#444", lw=0.8, rad=0.0, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=7, connectionstyle=f"arc3,rad={rad}", ls=ls, shrinkA=0, shrinkB=0))


def _canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


@figure("arch_mlp", "Arquitectura: MLP (LeNet-300-100)",
        "Cada barra es una capa de neuronas (alto proporcional al logaritmo del tamaño). Entre capas, los pesos que se podan (verde) y cuántos son. Casi todo está en la primera capa: 235k de 266k.")
def fig_arch_mlp():
    fig, ax = _canvas(6.5, 1.65, 10, 3.0)
    xs = [0.6, 3.1, 5.6, 8.1]
    sizes = [784, 300, 100, 10]
    names = ["Entrada\n784 = 28×28", "Oculta 1\n300 + ReLU", "Oculta 2\n100 + ReLU", "Salida\n10 clases"]
    hs = [0.55 * np.log10(n) + 0.15 for n in sizes]
    kinds = ["i", "a", "a", "i"]
    for x, h, n, k in zip(xs, hs, names, kinds):
        fc, ec = KIND[k]
        ax.add_patch(FancyBboxPatch((x, 1.55 - h / 2), 0.9, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=1))
        ax.text(x + 0.45, 0.45, n, ha="center", va="center", fontsize=7, linespacing=1.2)
    rng = np.random.default_rng(0)
    labels = ["Linear 784→300\n235,200 pesos", "Linear 300→100\n30,000 pesos", "Linear 100→10\n1,000 pesos"]
    for i in range(3):
        x0, x1 = xs[i] + 0.9, xs[i + 1]
        for _ in range(26):
            y0 = 1.55 + rng.uniform(-hs[i] / 2, hs[i] / 2) * 0.9
            y1 = 1.55 + rng.uniform(-hs[i + 1] / 2, hs[i + 1] / 2) * 0.9
            ax.plot([x0, x1], [y0, y1], color="#2a9d6f", lw=0.4, alpha=0.5)
        ax.text((x0 + x1) / 2, 2.85, labels[i], ha="center", va="center", fontsize=7, color="#1d7a54", fontweight="bold", linespacing=1.2)
    save(fig, "arch_mlp"); plt.close(fig)


@figure("arch_cnn", "Arquitectura: CNN (Conv-4)",
        "Cada pila es un tensor: más profundas = más canales, más altas = mayor resolución. Los números verdes son los pesos de cada capa; la primera capa lineal (2.1M) concentra el 86% del total.")
def fig_arch_cnn():
    fig, ax = _canvas(6.5, 2.85, 13, 4.55)
    base = 1.35
    def stack(x, spatial_h, depth_n, label, sub=None, color="i"):
        fc, ec = KIND[color]
        for k in range(depth_n):
            ax.add_patch(FancyBboxPatch((x + 0.09 * k, base + 0.09 * k), 0.38, spatial_h, boxstyle="round,pad=0,rounding_size=0.03", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 0.2 + 0.045 * depth_n, base - 0.22, label, ha="center", va="top", fontsize=6.9, linespacing=1.15)
    def bar(x, hgt, label, color="a"):
        fc, ec = KIND[color]
        ax.add_patch(FancyBboxPatch((x, base), 0.38, hgt, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 0.19, base - 0.22, label, ha="center", va="top", fontsize=6.9, linespacing=1.15)
    H32, H16, H8 = 1.5, 0.95, 0.6
    pos = [0.3, 1.6, 2.9, 4.2, 5.5, 6.8, 8.1]
    stack(pos[0], H32, 1, "imagen\n3×32×32")
    stack(pos[1], H32, 3, "64×32×32")
    stack(pos[2], H32, 3, "64×32×32")
    stack(pos[3], H16, 3, "64×16×16\n(pool)")
    stack(pos[4], H16, 4, "128×16×16")
    stack(pos[5], H16, 4, "128×16×16")
    stack(pos[6], H8, 4, "128×8×8\n(pool)")
    fcx = [9.55, 10.6, 11.65, 12.45]
    bar(fcx[0], 1.6, "8192\n(aplanar)", "n")
    bar(fcx[1], 0.8, "256\n+ReLU", "a")
    bar(fcx[2], 0.8, "256\n+ReLU", "a")
    bar(fcx[3] + 0.0, 0.2, "10", "i")
    # etiquetas de operaciones con pesos
    ops = [(pos[0] + 0.55, pos[1] - 0.05, "conv1", "1.7k"), (pos[1] + 0.55, pos[2] - 0.05, "conv2", "37k"), (pos[3] + 0.55, pos[4] - 0.05, "conv3", "74k"), (pos[4] + 0.55, pos[5] - 0.05, "conv4", "147k")]
    for x0, x1, nm, p in ops:
        _arr(ax, x0 + 0.02, base + 0.75, x1, base + 0.75, color="#2a9d6f", lw=1.1)
        ax.text((x0 + x1) / 2 + 0.05, base + 2.0, f"{nm}\n3×3", ha="center", fontsize=7.0, va="bottom", linespacing=1.1)
        ax.text((x0 + x1) / 2 + 0.05, base + 1.98, p, ha="center", fontsize=7.0, va="top", color="#1d7a54", fontweight="bold")
    for xa, xb in [(pos[2] + 0.5, pos[3] - 0.05), (pos[5] + 0.55, pos[6] - 0.05)]:
        _arr(ax, xa, base + 0.65, xb, base + 0.65, color="#888")
        ax.text((xa + xb) / 2, base + 0.78, "pool", ha="center", fontsize=6.6, color="#666")
    _arr(ax, pos[6] + 0.55, base + 0.3, fcx[0] - 0.05, base + 0.3, color="#888")
    for i, (nm, p) in enumerate([("fc1", "2.1M"), ("fc2", "66k"), ("fc3", "2.6k")]):
        xa, xb = fcx[i] + 0.4, fcx[i + 1] - 0.02
        _arr(ax, xa, base + 0.45, xb, base + 0.45, color="#2a9d6f", lw=1.1)
        ax.text((xa + xb) / 2 + 0.02, base + 1.85, nm, ha="center", fontsize=7.0, va="bottom")
        ax.text((xa + xb) / 2 + 0.02, base + 1.83, p, ha="center", fontsize=7.0, va="top", color="#1d7a54", fontweight="bold")
    ax.text(0.3, 4.3, "Conv-4: 4 convoluciones + 3 capas lineales, ~2.4M pesos podables", fontsize=8, fontweight="bold", va="center")
    ax.text(0.3, 0.2, LEGEND.split(" · ")[0] + ". Los números sobre cada flecha son los pesos de esa capa.", fontsize=6.9, color="#555", va="center")
    save(fig, "arch_cnn"); plt.close(fig)


@figure("arch_resnet", "Arquitectura: ResNet de 20 capas",
        "Arriba, la red completa: una convolución inicial, tres etapas de tres bloques residuales (los arcos son los atajos) y una capa final que no se poda. Abajo, el interior de un bloque.")
def fig_arch_resnet():
    fig, ax = _canvas(6.5, 2.7, 13, 5.5)
    ax.text(0.1, 5.3, "ResNet-20: 1 conv + 9 bloques residuales (2 conv cada uno) + 1 lineal, ~270k pesos podables", fontsize=8, fontweight="bold", va="center")
    y, h = 3.6, 0.85
    _blk(ax, 0.1, y, 0.95, h, "imagen\n3×32×32", "i")
    _blk(ax, 1.3, y, 1.1, h, "conv 3×3\n3→16\n+BN+ReLU", "p")
    _arr(ax, 1.07, y + h / 2, 1.28, y + h / 2)
    x = 2.8
    stages = [("16 canales · 32×32", 16), ("32 canales · 16×16 (÷2)", 32), ("64 canales · 8×8 (÷2)", 64)]
    prev = 2.42
    for si, (lab, c) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x - 0.1, y - 0.25), 2.9, h + 0.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#bbb", lw=0.8, ls="--"))
        ax.text(x + 1.35, y + h + 0.45, lab, ha="center", fontsize=6.6, color="#555")
        _arr(ax, prev, y + h / 2, x - 0.02, y + h / 2)
        for b in range(3):
            bx = x + b * 0.98
            _blk(ax, bx, y, 0.8, h, "bloque", "p", fs=6.3)
            if b < 2:
                _arr(ax, bx + 0.8, y + h / 2, bx + 0.97, y + h / 2)
            _arr(ax, bx + 0.1, y + h + 0.02, bx + 0.7, y + h + 0.02, color="#d97f00", rad=-0.8, lw=0.7)
        prev = x + 2 * 0.98 + 0.8
        x += 3.05
    _blk(ax, 11.7, y, 1.25, h, "AvgPool +\nLinear\n64→10", "n", fs=6.3)
    _arr(ax, prev, y + h / 2, 11.68, y + h / 2)
    # interior de un bloque
    ax.text(0.1, 2.45, "Dentro de un bloque residual", fontsize=7.5, fontweight="bold", va="center")
    yy, hh = 1.3, 0.8
    parts = [("x", "i", 0.45), ("conv 3×3", "p", 1.15), ("BN", "n", 0.65), ("ReLU", "a", 0.8), ("conv 3×3", "p", 1.15), ("BN", "n", 0.65), ("+", "i", 0.5), ("ReLU", "a", 0.8)]
    xx = 0.8
    xs = []
    for t, k, w in parts:
        _blk(ax, xx, yy, w, hh, t, k, fs=6.8)
        xs.append((xx, w))
        xx += w + 0.45
    for (x0, w0), (x1, _) in zip(xs[:-1], xs[1:]):
        _arr(ax, x0 + w0, yy + hh / 2, x1, yy + hh / 2)
    _arr(ax, xs[0][0] + xs[0][1] / 2, yy, xs[6][0] + xs[6][1] / 2, yy, color="#d97f00", rad=0.17, lw=1.0)
    ax.text((xs[0][0] + xs[6][0]) / 2 + 1.0, 0.18, "atajo: se suma la entrada (con conv 1×1 si cambian las dimensiones)", ha="center", fontsize=6.3, color="#d97f00")
    ax.text(12.9, 1.4, "verde: se poda\ngris: sin poda\nnaranja: activación/atajo", ha="right", va="center", fontsize=6.3, color="#555", linespacing=1.4)
    save(fig, "arch_resnet"); plt.close(fig)


@figure("arch_vit", "Arquitectura: ViT pequeño",
        "Arriba, de la imagen a la predicción: la imagen se corta en 64 parches de 4×4, cada uno es un token; se añade un token [CLS] que al final se clasifica. Abajo, el interior de uno de los 6 bloques. Todas las capas lineales y las LayerNorm entran a la poda global.")
def fig_arch_vit():
    fig, ax = _canvas(6.5, 3.6, 13, 7.0)
    ax.text(0.1, 6.7, "ViT pequeño: 6 bloques, dimensión 192, 3 cabezas, ~1.8M pesos podables", fontsize=8, fontweight="bold", va="center")
    y0 = 4.05
    # imagen y parches
    for i in range(4):
        for j in range(4):
            ax.add_patch(plt.Rectangle((0.2 + i * 0.3, y0 + 0.2 + j * 0.3), 0.3, 0.3, fc="#dce8f7" if (i + j) % 2 else "#c4d8f0", ec="#1f77b4", lw=0.5))
    ax.text(0.8, y0 - 0.15, "imagen 32×32\n→ 64 parches 4×4", ha="center", va="top", fontsize=6.2, linespacing=1.15)
    _arr(ax, 1.55, y0 + 0.8, 2.2, y0 + 0.8)
    _blk(ax, 2.25, y0 + 0.4, 1.35, 0.8, "Conv 4×4\n(3→192)\n= embedding", "p", fs=6.3)
    _arr(ax, 3.62, y0 + 0.8, 4.2, y0 + 0.8)
    # tokens
    for k in range(9):
        ax.add_patch(plt.Rectangle((4.25 + k * 0.17, y0 + 0.3), 0.14, 1.0, fc="#fbd9a5" if k == 0 else "#e6dcf5", ec="#8e5fd0" if k else "#d97f00", lw=0.5))
    ax.text(4.25 + 0.07, y0 + 1.45, "[CLS]", ha="center", fontsize=5.8, color="#d97f00")
    ax.text(5.1, y0 - 0.15, "65 tokens × 192\n(+ posición)", ha="center", va="top", fontsize=6.2, linespacing=1.15)
    _arr(ax, 5.95, y0 + 0.8, 6.5, y0 + 0.8)
    # pila de bloques
    for k in range(3, -1, -1):
        _blk(ax, 6.55 + 0.1 * k, y0 + 0.3 + 0.1 * k, 1.5, 1.0, "" if k else "× 6 bloques\nde atención", "v", fs=6.6)
    _arr(ax, 8.3, y0 + 0.8, 8.85, y0 + 0.8)
    _blk(ax, 8.9, y0 + 0.4, 1.0, 0.8, "LayerNorm", "n", fs=6.4)
    _arr(ax, 9.92, y0 + 0.8, 10.35, y0 + 0.8)
    _blk(ax, 10.4, y0 + 0.4, 1.1, 0.8, "solo [CLS]", "i", fs=6.4)
    _arr(ax, 11.52, y0 + 0.8, 11.95, y0 + 0.8)
    _blk(ax, 12.0, y0 + 0.4, 0.95, 0.8, "Linear\n192→10", "p", fs=6.3)
    # interior del bloque
    ax.text(0.1, 3.0, "Dentro de un bloque (se repite 6 veces)", fontsize=7.5, fontweight="bold", va="center")
    yy, hh = 1.35, 0.8
    parts = [("x", "i", 0.4), ("Layer\nNorm", "n", 0.75), ("qkv\n192→576", "p", 0.9), ("Atención\n3 cabezas", "v", 0.95), ("proj\n192→192", "p", 0.9), ("+", "i", 0.4),
             ("Layer\nNorm", "n", 0.75), ("mlp.0\n192→384", "p", 0.9), ("GELU", "a", 0.6), ("mlp.2\n384→192", "p", 0.9), ("+", "i", 0.4)]
    xx, xs = 0.25, []
    for t, k, w in parts:
        _blk(ax, xx, yy, w, hh, t, k, fs=6.6)
        xs.append((xx, w)); xx += w + 0.2
    for (a0, aw), (b0, _) in zip(xs[:-1], xs[1:]):
        _arr(ax, a0 + aw, yy + hh / 2, b0, yy + hh / 2)
    _arr(ax, xs[0][0] + 0.2, yy + hh, xs[5][0] + 0.2, yy + hh, color="#d97f00", rad=-0.2, lw=0.9)
    _arr(ax, xs[5][0] + 0.2, yy, xs[10][0] + 0.2, yy, color="#d97f00", rad=0.2, lw=0.9)
    ax.text(0.25, 0.45, "arcos naranja: conexiones residuales (se suma la entrada)", fontsize=6.7, color="#d97f00", va="center")
    ax.text(12.9, 0.45, LEGEND.split(" · ")[0] + " · morado: atención", ha="right", fontsize=6.7, color="#555", va="center")
    save(fig, "arch_vit"); plt.close(fig)


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
        "Atención del token [CLS] hacia los 64 parches (8×8) en el último bloque, promediada sobre las 3 cabezas, para 5 imágenes de prueba (columnas). Cada fila es un modelo con 11.6% de los pesos; 'aleatorio' tiene la misma máscara que el ticket con otra inicialización. Más brillante = más atención. Describe dónde pone peso la atención, no qué usa el modelo: su correlación con la oclusión es ≈ 0 (ver vit_occlusion).")
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


def _plain(ax, xt, yt):
    """Ejes logarítmicos con números normales en vez de potencias de 10."""
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt]); ax.minorticks_off()
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}%" for v in yt])


@figure("magnitudes", "Tamaño de los pesos por capa y qué sobrevive a la poda global",
        "Panel izquierdo: mediana de |w| por capa en la CNN densa entrenada (eje log; el tamaño de la capa está sobre cada barra). Panel central: mediana de |w| contra el porcentaje de pesos vivos de esa capa en el ticket global de la CNN (10.7% del total): las capas con pesos grandes (pocas entradas por neurona) sobreviven más. Panel derecho: lo mismo para el ViT largo (11.6% del total); la relación también es positiva (head y patch_embed tienen los pesos más grandes y sobreviven más), pero es menos limpia porque las magnitudes cambian durante las rondas de IMP y aquí se usan las del modelo denso.")
def fig_magnitudes():
    # CNN: pesos densos entrenados (02b) y máscara del ticket global con 10.7%
    base = torch.load(ROOT / "checkpoints" / "02b_cnn_global" / "base.pt", map_location="cpu")
    ck = torch.load(next((ROOT / "checkpoints" / "02b_cnn_global").glob("ticket_r10_*.pt")), map_location="cpu")
    med = {k: float(base["dense"][k + ".weight"].abs().median()) for k in CNN_LAYERS}
    size = {k: base["dense"][k + ".weight"].numel() for k in CNN_LAYERS}
    keep = {k: ck["masks"][k + ".weight"].float().mean().item() * 100 for k in CNN_LAYERS}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax = axes[0]
    ax.bar(range(7), [med[k] for k in CNN_LAYERS], color="#1f77b4")
    for i, k in enumerate(CNN_LAYERS):
        ax.text(i, med[k] * 1.12, f"{size[k]:,}", ha="center", fontsize=6.5, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=0.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(CNN_NICE, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mediana de |w| (red densa)")
    ax.set_title("CNN: tamaño de los pesos por capa\n(sobre cada barra, el número de pesos)", fontsize=8.5)
    ax = axes[1]
    for k, nm in zip(CNN_LAYERS, CNN_NICE):
        ax.scatter(med[k], keep[k], s=30, color="#ff7f0e")
        ax.annotate(nm.split(" ")[0], (med[k], keep[k]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mediana de |w| (red densa)")
    ax.set_ylabel("% vivo de la capa (ticket global, 10.7%)")
    ax.set_title("CNN: tamaño de los pesos vs. % que sobrevive", fontsize=8.5)
    _plain(ax, [0.01, 0.02, 0.05, 0.1], [5, 10, 20, 50, 100])
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
    ax.set_title("ViT largo: tamaño de los pesos vs. % que sobrevive", fontsize=8.5)
    _plain(ax, [0.04, 0.05, 0.06], [8, 10, 20, 40])
    fig.tight_layout()
    save(fig, "magnitudes")


@figure("vit_depth", "ViT largo: pesos vivos por bloque y tipo de capa",
        "Fracción de pesos vivos de cada capa dentro de cada uno de los 6 bloques (columna 1 = el más cercano a la entrada), con 11.6% y con 3.2% de los pesos totales. Clara = casi todos vivos. Muestra si la poda golpea más a los bloques tempranos o a los tardíos.")
def fig_vit_depth():
    c = vit_ctx()
    rows = {"11.6%": c["at"]("ticket"), "3.2%": c["curves"]["ticket"][-1]}
    types = ["qkv", "proj", "mlp.0", "mlp.2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    for ax, (lab, r) in zip(axes, rows.items()):
        ms = c["load"](r[3])["masks"]
        g = np.array([[ms[f"blocks.{i}.{t}.weight"].float().mean().item() * 100 for i in range(6)] for t in types])
        ax.grid(False)
        im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=0, vmax=max(g.max(), 1))
        ax.set_yticks(range(4))
        ax.set_yticklabels(types)
        ax.set_xticks(range(6))
        ax.set_xticklabels([str(i + 1) for i in range(6)])
        ax.set_xlabel("Bloque (1 = más cerca de la entrada)")
        ax.set_title(f"{lab} de los pesos totales", fontsize=9)
        for i in range(4):
            for j in range(6):
                ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color="w" if g[i, j] < g.max() * 0.55 else "k")
        fig.colorbar(im, ax=ax, label="% vivo")
    fig.tight_layout()
    save(fig, "vit_depth")


@figure("mlp_mask", "MLP: qué píxeles de la imagen sobreviven en la primera capa",
        "Para cada píxel de la imagen de entrada de 28×28, la fracción de las 300 neuronas ocultas que sigue conectada a él, en tres niveles de poda (por capa, 20% por ronda). Oscuro = el píxel casi no se usa. El borde de la imagen, que casi siempre vale 0 en MNIST, es lo primero que se desconecta.")
def fig_mlp_mask():
    d = ROOT / "checkpoints" / "01_mlp"
    picks = [("ticket_r04_", "41.0%"), ("ticket_r10_", "10.7%"), ("ticket_r15_", "3.5%")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, (prefix, lab) in zip(axes, picks):
        ck = torch.load(next(d.glob(prefix + "*.pt")), map_location="cpu")
        m = ck["masks"]["layers.0.weight"].float()
        per_pixel = m.mean(0).reshape(28, 28).numpy()
        dead = int((m.sum(0) == 0).sum())
        ax.grid(False)
        im = ax.imshow(per_pixel, cmap="viridis", vmin=0, vmax=max(per_pixel.max(), 1e-6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{lab} de los pesos\n{dead} píxeles desconectados", fontsize=8.5)
    fig.colorbar(im, ax=axes, label="fracción de neuronas conectadas", shrink=0.8)
    save(fig, "mlp_mask")


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
    # similitud entre los mapas de distintos modelos (presentation/map_similarity.py)
    sim = ROOT / "results" / "map_similarity.json"
    if sim.exists():
        d = json.loads(sim.read_text())["pares"]
        nm = {"denso": "denso", "ticket": "ticket", "rewinding": "rewinding", "aleatorio": "control aleatorio"}
        order = ["denso vs ticket", "denso vs rewinding", "ticket vs rewinding", "denso vs aleatorio", "ticket vs aleatorio", "rewinding vs aleatorio"]
        rows = []
        for k in order:
            a, b = k.split(" vs ")
            rows.append(f"{nm[a]} -- {nm[b]} & {d[k]['att']['todas']:.2f} & {d[k]['occ']['todas']:.2f} \\\\")
            if k == "ticket vs rewinding":
                rows.append("\\midrule")
        (OUT / "table_mapsim.tex").write_text("\\begin{tabular}{@{}lrr@{}}\\toprule\nPar de modelos & Atenci\\'on & Oclusi\\'on \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
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
    ap.add_argument("--dark", action="store_true", help="solo los diagramas de arquitectura, en tema oscuro, a presentation/img/")
    args = ap.parse_args()
    if args.dark and not args.only:
        args.only = ["arch_mlp", "arch_cnn", "arch_resnet", "arch_vit"]
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
