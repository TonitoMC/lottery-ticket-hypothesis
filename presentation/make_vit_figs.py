"""Figuras de la diapositiva del ViT: diagrama con pesos vivos por capa y atención sobre imágenes reales.

    cd <raíz> && uv run python presentation/make_vit_figs.py

Escribe presentation/img/vit_diagram.png y presentation/img/vit_attn.png.
Reutiliza el modelo y los cargadores de report/make_figures.py. El código de las figuras fue generado con ayuda de IA.
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
sys.path.insert(0, str(root / "report"))
import make_figures as mf  # noqa: E402

LIGHT = "--light" in sys.argv
if LIGHT:
    OUT = root / "report" / "figures"
    BG, INK, MUTED, LINE, BOX = "#ffffff", "#111111", "#555555", "#c8cde0", "#eef2fb"
    BLUE, LILAC, GREEN, ORANGE = "#1f77b4", "#8e5fd0", "#2a9d6f", "#d97f00"
else:
    OUT = Path(__file__).parent / "img"
    BG, INK, MUTED, LINE, BOX = "#1c2140", "#f2f3f8", "#a3acc9", "#2b3157", "#232a5a"
    BLUE, LILAC, GREEN, ORANGE = "#6ea8ff", "#c9a7ff", "#3ddc97", "#ffb454"

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "font.size": 10, "axes.grid": False})

c = mf.vit_ctx()

# ------------------------------------------------------------------ diagrama
def alive_by_type(path):
    ms = c["load"](path)["masks"]
    out = {}
    for n, m in ms.items():
        if m.dim() >= 2:
            t = mf.layer_type(n)
            a, tot = out.get(t, (0, 0))
            out[t] = (a + int(m.sum()), tot + m.numel())
    return {t: (a / tot * 100, a, tot) for t, (a, tot) in out.items()}


r116 = c["at"]("ticket")      # ~11.6%
r32 = c["curves"]["ticket"][-1]  # 3.2%
A, B = alive_by_type(r116[3]), alive_by_type(r32[3])
meta = {"patch_embed": ("patch_embed", "Conv 3→192, parche 4×4"), "qkv": ("qkv", "192→576"), "proj": ("proj", "192→192"),
        "mlp.0": ("mlp.0", "192→384"), "mlp.2": ("mlp.2", "384→192"), "head": ("head", "192→10")}
order = ["patch_embed", "qkv", "proj", "mlp.0", "mlp.2", "head"]
xs = {"patch_embed": 0.2, "qkv": 2.55, "proj": 4.0, "mlp.0": 5.45, "mlp.2": 6.9, "head": 9.25}
bw = 1.3
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.25)
ax.axis("off")
# agrupación de los 6 bloques
ax.add_patch(FancyBboxPatch((2.35, 0.55), 6.1, 3.85, boxstyle="round,pad=0.02", fc="none", ec=MUTED, lw=1, ls="--"))
ax.text(5.4, 4.28, "× 6 bloques de atención  (pesos sumados sobre los 6)", ha="center", va="center", fontsize=8.5, color=MUTED)
for name in order:
    x = xs[name]
    la, tot = A[name][0], A[name][2]
    ax.add_patch(FancyBboxPatch((x, 3.2), bw, 0.8, boxstyle="round,pad=0.02", fc=BOX, ec=BLUE, lw=1.2))
    ax.text(x + bw / 2, 3.78, name, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + bw / 2, 3.45, f"{meta[name][1]}\n{tot:,} pesos", ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.15)
    for j, (D, col) in enumerate(((A, BLUE), (B, LILAC))):
        bx = x + 0.1 + j * 0.62
        ax.add_patch(Rectangle((bx, 0.95), 0.5, 2.0, fc="none", ec=LINE, lw=0.8))
        h = 2.0 * D[name][0] / 100
        ax.add_patch(Rectangle((bx, 0.95), 0.5, max(h, 0.03), fc=col, ec="none", alpha=0.92))
        v = D[name][0]
        ax.text(bx + 0.25, 0.95 + max(h, 0.03) + 0.06, f"{v:.0f}%" if v >= 9.5 else f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
        ax.text(bx + 0.25, 0.8, f"{D[name][1]:,}", ha="center", va="top", fontsize=6.2, color=col)
ax.annotate("", xy=(2.3, 3.6), xytext=(1.55, 3.6), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
ax.annotate("", xy=(9.2, 3.6), xytext=(8.5, 3.6), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
ax.text(0.2, 5.05, "ViT largo: cuántos pesos sobreviven en cada tipo de capa (poda global)", fontsize=12.5, fontweight="bold")
ax.text(0.2, 4.76, "Barra = % de pesos vivos de ese tipo de capa; número de abajo = pesos que quedan", fontsize=8.5, color=MUTED)
ax.add_patch(Rectangle((7.2, 4.68), 0.2, 0.18, fc=BLUE)); ax.text(7.48, 4.77, f"{r116[1]:.1f}% de pesos en total", va="center", fontsize=8.5, color=BLUE)
ax.add_patch(Rectangle((9.5, 4.68), 0.2, 0.18, fc=LILAC)); ax.text(9.78, 4.77, f"{r32[1]:.1f}%", va="center", fontsize=8.5, color=LILAC)
ax.text(0.15, 0.25, "Sobreviven más la entrada (patch_embed) y la salida (head); las capas MLP internas son las más podadas.", fontsize=8.8, color=INK)
fig.savefig(OUT / "vit_diagram.png", dpi=190, bbox_inches="tight")
if LIGHT: fig.savefig(OUT / "vit_diagram.pdf", bbox_inches="tight")
plt.close(fig)
print("capas 11.6%:", {k: round(v[0], 1) for k, v in A.items()}, "| 3.2%:", {k: round(v[0], 1) for k, v in B.items()})

# ------------------------------------------------------------------ atención sobre imágenes
test, mean, std = c["test"], c["mean"], c["std"]
models = {"denso": mf.vit(c["base"]["dense"]),
          "ticket": mf.vit(c["load"](c["at"]("ticket")[3])["model"]),
          "rewinding": mf.vit(c["load"](c["at"]("late")[3])["model"]),
          "aleatorio": mf.vit(c["load"](c["at"]("random")[3])["model"])}
N = 1500
X = torch.stack([mf.img_tensor(test, i) for i in range(N)])
Y = torch.tensor(test.targets[:N])
with torch.no_grad():
    pred = {k: torch.cat([m((X[i:i + 250] - mean) / std).argmax(1) for i in range(0, N, 250)]) for k, m in models.items()}
ok = (pred["denso"] == Y) & (pred["ticket"] == Y) & (pred["rewinding"] == Y) & (pred["aleatorio"] != Y)
cand = ok.nonzero().flatten().tolist()
chosen, seen = [], set()
for i in cand:
    if int(Y[i]) not in seen:
        chosen.append(i)
        seen.add(int(Y[i]))
    if len(chosen) == 3:
        break
print("imágenes elegidas:", chosen, [test.classes[int(Y[i])] for i in chosen], f"({len(cand)} candidatas de {N})")
names = test.classes
fig, axes = plt.subplots(len(chosen), 5, figsize=(10.5, 2.15 * len(chosen)))
for r, i in enumerate(chosen):
    x = ((X[i:i + 1] - mean) / std)
    ax = axes[r, 0]
    ax.imshow(X[i].permute(1, 2, 0), interpolation="nearest")
    ax.set_title(f"real: {names[int(Y[i])]}" if r == 0 else names[int(Y[i])], fontsize=9, color=INK)
    ax.axis("off")
    for k, (n, m) in enumerate(models.items(), start=1):
        with torch.no_grad():
            a = m.all_attention(x)[-1][0, :, 0, 1:].mean(0).reshape(8, 8)
            p = int(m(x).argmax(1))
        ax = axes[r, k]
        ax.imshow(X[i].permute(1, 2, 0), interpolation="nearest")
        ax.imshow(a, extent=(-0.5, 31.5, 31.5, -0.5), cmap="inferno", alpha=0.62, interpolation="bilinear")
        good = p == int(Y[i])
        ax.set_title((f"{n}\n" if r == 0 else "") + f"{'✓' if good else '✗'} {names[p]}", fontsize=9, color=GREEN if good else ORANGE)
        ax.axis("off")
fig.suptitle(f"Atención del [CLS] y predicción con 11.6% de los pesos (brillante = más atención).\nImágenes ELEGIDAS: {len(cand)} de {N} donde denso, ticket y rewind aciertan y el aleatorio falla; no son representativas", fontsize=9, color=MUTED, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "vit_attn.png", dpi=190)
if LIGHT: fig.savefig(OUT / "vit_attn.pdf")
plt.close(fig)
print("listo")
