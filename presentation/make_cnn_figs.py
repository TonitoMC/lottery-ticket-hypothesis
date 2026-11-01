"""Figuras de la diapositiva de la CNN (poda por capa vs. global), con los checkpoints reales.

    cd <raíz> && uv run python presentation/make_cnn_figs.py

Escribe presentation/img/cnn_diagram.png y presentation/img/cnn_filters.png.
El código de las figuras fue generado con ayuda de IA; los pesos y las máscaras son los de las corridas.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

root = Path(__file__).resolve().parent.parent

LIGHT = "--light" in sys.argv
if LIGHT:
    OUT = root / "report" / "figures"
    BG, INK, MUTED, LINE, BOX = "#ffffff", "#111111", "#555555", "#c8cde0", "#eef2fb"
    BLUE, LILAC, GREEN, ORANGE = "#1f77b4", "#8e5fd0", "#2a9d6f", "#d97f00"
else:
    OUT = Path(__file__).parent / "img"
    BG, INK, MUTED, LINE, BOX = "#1c2140", "#f2f3f8", "#a3acc9", "#2b3157", "#232a5a"
    BLUE, LILAC, GREEN, ORANGE = "#6ea8ff", "#c9a7ff", "#3ddc97", "#ffb454"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "font.size": 10})

LAYERS = ["conv.0", "conv.2", "conv.5", "conv.7", "linear.1", "linear.3", "linear.5"]
NICE = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2", "fc3"]
SHAPES = ["3→64", "64→64", "64→128", "128→128", "8192→256", "256→256", "256→10"]


def load(d):
    ck = torch.load(next((root / "checkpoints" / d).glob("ticket_r10_*.pt")), map_location="cpu")
    base = torch.load(root / "checkpoints" / d / "base.pt", map_location="cpu")
    return ck, base


ck_l, base_l = load("02_cnn")
ck_g, base_g = load("02b_cnn_global")
keep = {k: [ck[ "masks"][n + ".weight"].float().mean().item() * 100 for n in LAYERS] for k, ck in (("layer", ck_l), ("global", ck_g))}
counts = {k: [int(ck["masks"][n + ".weight"].sum()) for n in LAYERS] for k, ck in (("layer", ck_l), ("global", ck_g))}
total = [ck_g["masks"][n + ".weight"].numel() for n in LAYERS]

# ------------------------------------------------------------------ diagrama
fig, ax = plt.subplots(figsize=(13, 4.9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.9)
ax.axis("off")
slots = [("conv1", 0.2), ("conv2", 1.7), ("pool", 3.05), ("conv3", 3.9), ("conv4", 5.4), ("pool", 6.75), ("fc1", 7.6), ("fc2", 9.1), ("fc3", 10.6)]
li = {n: i for i, n in enumerate(NICE)}
bw = 1.15
for name, x in slots:
    if name == "pool":
        ax.add_patch(FancyBboxPatch((x, 3.35), 0.55, 0.75, boxstyle="round,pad=0.02", fc=LINE, ec=MUTED, lw=0.8))
        ax.text(x + 0.275, 3.725, "pool\n÷2", ha="center", va="center", fontsize=7.5, color=MUTED)
        continue
    i = li[name]
    ax.add_patch(FancyBboxPatch((x, 3.35), bw, 0.75, boxstyle="round,pad=0.02", fc=BOX, ec=BLUE, lw=1.2))
    ax.text(x + bw / 2, 3.88, name, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + bw / 2, 3.58, f"{SHAPES[i]}\n{total[i]:,} pesos", ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.15)
    # barras de supervivencia: por capa (naranja) y global (verde)
    for j, (key, col) in enumerate((("layer", ORANGE), ("global", GREEN))):
        bx = x + 0.1 + j * 0.55
        ax.add_patch(Rectangle((bx, 0.9), 0.45, 2.1, fc="none", ec=LINE, lw=0.8))
        h = 2.1 * keep[key][i] / 100
        ax.add_patch(Rectangle((bx, 0.9), 0.45, max(h, 0.03), fc=col, ec="none", alpha=0.9))
        ax.text(bx + 0.225, 0.9 + max(h, 0.03) + 0.07, f"{keep[key][i]:.0f}%" if keep[key][i] >= 9.5 else f"{keep[key][i]:.1f}%", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")
        ax.text(bx + 0.225, 0.72, f"{counts[key][i]:,}", ha="center", va="top", fontsize=6.3, color=col)
    ax.annotate("", xy=(x + bw + 0.12, 3.72), xytext=(x + bw + 0.01, 3.72), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8)) if name not in ("conv2", "conv4", "fc3") else None
ax.text(0.2, 4.62, "CNN (Conv-4) con 10.7% de los pesos: cuántos pesos sobreviven en cada capa", fontsize=12.5, fontweight="bold")
ax.text(0.2, 4.35, "Barra = % de pesos vivos de esa capa; número de abajo = pesos que quedan", fontsize=8.5, color=MUTED)
ax.add_patch(Rectangle((9.7, 4.38), 0.2, 0.18, fc=ORANGE)); ax.text(9.98, 4.47, "poda por capa (02)", va="center", fontsize=8.5, color=ORANGE)
ax.add_patch(Rectangle((11.4, 4.38), 0.2, 0.18, fc=GREEN)); ax.text(11.68, 4.47, "poda global (02b)", va="center", fontsize=8.5, color=GREEN)
ax.text(0.15, 0.25, "Por capa: cada capa pierde 89%, incluso conv1 (1,728 pesos).", fontsize=8.5, color=ORANGE)
ax.text(5.1, 0.25, "Global: conv1 conserva 88%; la capa enorme fc1 (2.1M) absorbe casi toda la poda.", fontsize=8.5, color=GREEN)
fig.savefig(OUT / "cnn_diagram.png", dpi=190, bbox_inches="tight")
if LIGHT: fig.savefig(OUT / "cnn_diagram.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ filtros de conv1
def filters(state):
    w = state["conv.0.weight"].clone()  # [64,3,3,3]
    return w


panels = [("Red densa entrenada", filters(base_g["dense"]), None),
          (f"Poda por capa, 10.7%", filters(ck_l["model"]), ck_l["masks"]["conv.0.weight"]),
          (f"Poda global, 10.7%", filters(ck_g["model"]), ck_g["masks"]["conv.0.weight"])]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
for ax, (title, w, m) in zip(axes, panels):
    scale = float(w.abs().max()) + 1e-9
    canvas = np.zeros((8 * 4, 8 * 4, 3))
    empty = 0
    for k in range(64):
        r, c = divmod(k, 8)
        f = w[k].permute(1, 2, 0).numpy()  # [3,3,3] (alto, ancho, canal)
        alive = (m[k].permute(1, 2, 0).numpy() > 0) if m is not None else np.ones_like(f, dtype=bool)
        if not alive.any():
            canvas[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = np.array([0.8, 0.15, 0.15])
            empty += 1
            continue
        tile = np.clip(0.5 + f / (2 * scale), 0, 1)
        tile[~alive] = 0.5            # canal podado: gris neutro
        tile[~alive.any(axis=2)] = 0  # posición sin ningún peso vivo: negro
        canvas[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = tile
    ax.imshow(canvas, interpolation="nearest")
    n_alive = int((m > 0).sum()) if m is not None else w.numel()
    ax.set_title(f"{title}\n{n_alive:,} de {w.numel():,} pesos · {empty} filtros vacíos", fontsize=9.5, color=GREEN if empty == 0 else ORANGE)
    ax.axis("off")
fig.suptitle("Los 64 filtros de la primera convolución (cada cuadro = un filtro 3×3×3). Negro = posición podada · rojo = filtro sin ningún peso vivo", fontsize=9.5, color=MUTED, y=0.035)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT / "cnn_filters.png", dpi=200)
if LIGHT: fig.savefig(OUT / "cnn_filters.pdf")
plt.close(fig)
print("listo", {k: [round(x, 1) for x in v] for k, v in keep.items()})
