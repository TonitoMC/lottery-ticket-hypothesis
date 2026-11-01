import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    ## 06 - Inspección de los tickets
    Los notebooks 01-05 guardan en `checkpoints/<corrida>/` los pesos y las máscaras de cada ronda de IMP (`lth.save_round`) y los pesos denso / inicial / de rewind (`base.pt`). Aquí se abren esos archivos para mirar **qué** encontró la poda, no solo cuánta accuracy queda:

    1. **Pesos que sobreviven por capa** (y por cabeza de atención en el ViT).
    2. **Accuracy por clase**, denso vs ticket vs rewinding vs random.
    3. **Mapas de atención** y su entropía.
    4. **Gradientes** por capa, al inicio del entrenamiento y al final.

    Las secciones 2-4 usan el ViT de 04/05 (la clase está definida aquí abajo). La sección 1 solo usa máscaras, así que funciona con cualquier carpeta de checkpoints.

    Las curvas se comparan siempre al mismo porcentaje de pesos (`TARGET_PCT`). Todo es de una sola corrida, así que diferencias pequeñas no se deben interpretar.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import os
    import json
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    import lth

    # Carpeta de checkpoints a inspeccionar y porcentaje de pesos con el que se comparan las curvas
    RUN_DIR = os.environ.get("INSPECT_DIR", "../checkpoints/05_vit_longer")
    TARGET_PCT = float(os.environ.get("INSPECT_PCT", 10.0))

    # Set GPU - CPU as device
    device = os.environ.get("INSPECT_DEVICE") or (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    print(f"Using {device} device, inspecting {RUN_DIR}")
    return F, RUN_DIR, TARGET_PCT, DataLoader, datasets, device, json, lth, mo, nn, np, os, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoints
    Se listan los archivos de la carpeta. Cada curva (`ticket`, `late`, `random`) tiene una entrada por ronda; la ronda 0 es el modelo denso.
    """)
    return


@app.cell
def _(RUN_DIR, TARGET_PCT, lth, os, torch):
    base = torch.load(os.path.join(RUN_DIR, "base.pt"), map_location="cpu")

    # curves[tag] = lista de (ronda, % de pesos, acc, ruta) ordenada por ronda
    curves = {}
    for _tag, _r, _pct, _acc, _path in lth.list_checkpoints(RUN_DIR):
        curves.setdefault(_tag, []).append((_r, _pct, _acc, _path))
    for _tag in curves:
        curves[_tag].sort()

    # Ronda de cada curva más cercana a TARGET_PCT
    def at_target(tag):
        return min(curves[tag], key=lambda c: abs(c[1] - TARGET_PCT))

    for _tag, _rows in curves.items():
        print(_tag)
        for _r, _pct, _acc, _ in _rows:
            print(f"  ronda {_r:2d}: {_pct:6.2f}% de pesos, acc = {_acc:.4f}")
    print("rewind en base.pt:", base["rewind"] is not None)

    # Máscaras y pesos de un checkpoint
    _cache = {}
    def ckpt(path):
        if path not in _cache:
            _cache[path] = torch.load(path, map_location="cpu")
        return _cache[path]
    return at_target, base, ckpt, curves


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pesos que sobreviven por capa
    Fracción de pesos que sigue viva en cada capa, para cada ronda del ticket. Con poda **global** (03, 04, 05) cada capa se poda distinto: las capas cuyos pesos son pequeños pierden más. Con poda **por capa** (01, 02) todas deberían bajar parecido, y cualquier capa que se desvíe es sospechosa (útil para revisar el punto de 10.7% de 02).

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(ckpt, curves, np, plt):
    _rows = curves["ticket"]
    _names = list(ckpt(_rows[0][3])["masks"].keys())
    # keep[i, j] = fracción viva de la capa i en la ronda j
    keep = np.array([[ckpt(p)["masks"][n].float().mean().item() for (_, _, _, p) in _rows] for n in _names])

    _fig, _ax = plt.subplots(figsize=(8, max(3, 0.28 * len(_names))))
    _im = _ax.imshow(keep, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    _ax.set_yticks(range(len(_names)))
    _ax.set_yticklabels(_names, fontsize=7)
    _ax.set_xticks(range(len(_rows)))
    _ax.set_xticklabels([f"{c[1]:.1f}%" for c in _rows], rotation=60, fontsize=7)
    _ax.set_xlabel("Pesos que quedan en total")
    _fig.colorbar(_im, label="Fracción viva de la capa")
    _fig.tight_layout()
    _fig
    return (keep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cabezas de atención (solo ViT)
    El peso `qkv` de cada bloque tiene 3 partes (q, k, v) con una fila de cabeza por cada cabeza de atención. Se muestra la fracción viva de cada parte y cabeza al porcentaje objetivo. Una cabeza con ~0% está efectivamente apagada.
    """)
    return


@app.cell
def _(at_target, ckpt, curves, np, plt):
    _r, _pct, _, _path = at_target("ticket")
    _m = ckpt(_path)["masks"]
    _qkv = [n for n in _m if n.endswith("qkv.weight")]
    if _qkv:
        _heads = 3
        # [3D, D] -> [3 (q,k,v), heads, D/heads, D]
        _grid = np.array([_m[n].float().reshape(3, _heads, -1, _m[n].shape[1]).mean(dim=(2, 3)).flatten().numpy() for n in _qkv])
        _fig, _ax = plt.subplots(figsize=(7, 3))
        _im = _ax.imshow(_grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        _ax.set_xticks(range(3 * _heads))
        _ax.set_xticklabels([f"{p}{h}" for p in "qkv" for h in range(_heads)])
        _ax.set_yticks(range(len(_qkv)))
        _ax.set_yticklabels([f"bloque {i}" for i in range(len(_qkv))])
        _ax.set_title(f"Fracción viva en qkv con {_pct:.1f}% de pesos (ronda {_r})")
        _fig.colorbar(_im)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = "Esta corrida no tiene capas qkv."
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ticket vs rewinding: ¿encuentran la misma estructura?
    Intersección sobre unión (IoU) de las máscaras del ticket clásico y del ticket con rewinding, por capa, al mismo porcentaje objetivo. IoU = 1 es la misma máscara; el IoU de dos máscaras aleatorias con la misma densidad $d$ sería $d/(2-d)$, que sirve de referencia.
    """)
    return


@app.cell
def _(at_target, ckpt, curves, plt):
    if "late" in curves:
        _a, _b = ckpt(at_target("ticket")[3])["masks"], ckpt(at_target("late")[3])["masks"]
        _names = list(_a.keys())
        _iou, _ref = [], []
        for _n in _names:
            _inter = (_a[_n] & _b[_n]).sum().item()
            _union = (_a[_n] | _b[_n]).sum().item()
            _iou.append(_inter / max(_union, 1))
            _d = _a[_n].float().mean().item()
            _ref.append(_d / (2 - _d))
        _fig, _ax = plt.subplots(figsize=(8, 3.5))
        _ax.bar(range(len(_names)), _iou, label="IoU ticket vs rewinding")
        _ax.plot(range(len(_names)), _ref, "k.", label="IoU esperado de dos máscaras aleatorias")
        _ax.set_xticks(range(len(_names)))
        _ax.set_xticklabels(_names, rotation=90, fontsize=6)
        _ax.legend()
        _fig.tight_layout()
        _out2 = _fig
    else:
        _out2 = "Esta corrida no tiene curva con rewinding."
    _out2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forma de las máscaras
    La fracción viva por capa no dice **dónde** quedaron los pesos. Aquí se mira la estructura de la máscara del ticket al porcentaje objetivo:

    - **Capas convolucionales** (`[filtros, canales, alto, ancho]`): qué posiciones del kernel sobreviven y cuántos filtros quedaron completamente vacíos (un filtro vacío es una unidad muerta).
    - **Capas densas** (`[salidas, entradas]`): cuántas neuronas de salida o de entrada quedaron sin ningún peso. Si la capa lee una imagen de 784 entradas (MLP), la fracción viva por entrada se dibuja como imagen de 28x28.
    - El histograma muestra cuántos pesos conserva cada unidad de salida. Poda no estructurada que reparte parejo da un pico alrededor de la fracción global; unidades casi vacías o casi llenas indican poda concentrada.
    """)
    return


@app.cell
def _(at_target, ckpt, mo):
    _r, _pct, _, _path = at_target("ticket")
    masks_t = ckpt(_path)["masks"]
    _rows = ["| Capa | Forma | Pesos vivos | Unidades de salida vacías | Entradas vacías |", "|---|---|---:|---:|---:|"]
    for _n, _m in masks_t.items():
        _alive = _m.float().mean().item()
        _out = _m.reshape(_m.shape[0], -1).any(dim=1)
        _inp = _m.transpose(0, 1).reshape(_m.shape[1], -1).any(dim=1)
        _rows.append(f"| `{_n}` | {tuple(_m.shape)} | {int(_m.sum())} / {_m.numel()} ({_alive:.1%}) | {int((~_out).sum())} / {len(_out)} | {int((~_inp).sum())} / {len(_inp)} |")
    mo.md(f"**Ticket con {_pct:.1f}% de los pesos (ronda {_r})**\n\n" + "\n".join(_rows))
    return (masks_t,)


@app.cell
def _(masks_t, np, plt):
    _names = list(masks_t.keys())
    _fig, _axes = plt.subplots(len(_names), 2, figsize=(9, 2.3 * len(_names)), squeeze=False)
    for _i, _n in enumerate(_names):
        _m = masks_t[_n].float()
        _a, _b = _axes[_i]
        if _m.dim() == 4:
            # Fracción viva por posición del kernel
            _k = _m.mean(dim=(0, 1)).numpy()
            _a.imshow(_k, vmin=0, vmax=1, cmap="viridis")
            for (_y, _x), _v in np.ndenumerate(_k):
                _a.text(_x, _y, f"{_v:.2f}", ha="center", va="center", fontsize=7, color="w")
            _a.set_title(f"{_n}: posiciones del kernel", fontsize=8)
        elif _m.shape[1] == 784:
            _a.imshow(_m.mean(dim=0).reshape(28, 28).numpy(), vmin=0, vmax=1, cmap="viridis")
            _a.set_title(f"{_n}: fracción viva por pixel de entrada", fontsize=8)
        else:
            _a.plot(np.sort(_m.mean(dim=0).numpy()))
            _a.set_ylim(0, 1)
            _a.set_title(f"{_n}: fracción viva por entrada (ordenada)", fontsize=8)
        _per_unit = _m.reshape(_m.shape[0], -1).mean(dim=1).numpy()
        _b.hist(_per_unit, bins=30, range=(0, 1))
        _b.axvline(_m.mean().item(), color="r", lw=1)
        _b.set_title(f"{_n}: pesos vivos por unidad de salida (línea roja: promedio de la capa)", fontsize=8)
        _a.tick_params(labelsize=7); _b.tick_params(labelsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ¿Los sobrevivientes ya eran grandes en la inicialización?
    La poda conserva los pesos de mayor magnitud **después de entrenar**. Aquí se compara el valor **inicial** $|\theta_0|$ de los pesos que sobrevivieron con el de los podados, por capa. Un cociente cercano a 1 significa que la inicialización casi no predice qué sobrevive (el entrenamiento decide); mayor a 1, que los pesos que ya empezaban grandes tienen más probabilidad de quedarse.
    """)
    return


@app.cell
def _(base, masks_t, plt):
    _names = list(masks_t.keys())
    _ratio = []
    for _n in _names:
        _w0 = base["init"][_n].abs()
        _m = masks_t[_n]
        _ratio.append((_w0[_m].mean() / _w0[~_m].mean()).item())
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.bar(range(len(_names)), _ratio)
    _ax.axhline(1, color="k", lw=0.8)
    _ax.set_xticks(range(len(_names)))
    _ax.set_xticklabels(_names, rotation=90, fontsize=7)
    _ax.set_ylabel("media |θ₀| sobrevivientes / podados")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelo y datos (ViT)
    La arquitectura es la de 04/05. Se evalúa con el conjunto de prueba completo, sin aumento de datos y en modo evaluación. Para los mapas de atención se usan 10 imágenes fijas, una por clase.
    """)
    return


@app.cell
def _(F, nn, torch):
    class Block(nn.Module):
        def __init__(self, dim, heads, mlp_dim):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

        # Pesos de atención [B, heads, N, N] (SDPA no los regresa)
        def attention(self, x):
            B, N, D = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            return torch.softmax(q @ k.transpose(-2, -1) / (D // self.heads) ** 0.5, dim=-1), v

        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, k, v)
            x = x + self.proj(attn.transpose(1, 2).reshape(B, N, D))
            return x + self.mlp(self.norm2(x))

    class ViT(nn.Module):
        def __init__(self, dim=192, depth=6, heads=3, mlp_dim=384, patch=4):
            super().__init__()
            n_patches = (32 // patch) ** 2
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
            self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 10)

        def embed(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            return torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos

        def forward(self, x):
            x = self.norm(self.blocks(self.embed(x)))
            return self.head(x[:, 0])

        # Pesos de atención de todos los bloques: lista de [B, heads, N, N]
        def all_attention(self, x):
            x, out = self.embed(x), []
            for blk in self.blocks:
                attn, _ = blk.attention(x)
                out.append(attn)
                x = blk(x)
            return out
    return (ViT,)


@app.cell
def _(DataLoader, datasets, torch, v2):
    _mean, _std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    cifar_test = datasets.CIFAR10(
        root="data", train=False, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(_mean, _std)]),
    )
    test_loader = DataLoader(cifar_test, 256, shuffle=False, num_workers=2)

    # Una imagen por clase, fijas para todas las comparaciones
    _targets = torch.tensor(cifar_test.targets)
    fixed_idx = [(_targets == c).nonzero()[0].item() for c in range(10)]
    fixed_x = torch.stack([cifar_test[i][0] for i in fixed_idx])
    # 512 imágenes para promedios (entropía, gradientes)
    sample_x = torch.stack([cifar_test[i][0] for i in range(512)])
    sample_y = torch.tensor(cifar_test.targets[:512])
    classes = cifar_test.classes
    mean, std = torch.tensor(_mean).view(3, 1, 1), torch.tensor(_std).view(3, 1, 1)
    return classes, fixed_x, mean, sample_x, sample_y, std, test_loader


@app.cell
def _(ViT, device, test_loader, torch):
    def load_vit(state):
        _m = ViT()
        _m.load_state_dict(state)
        return _m.to(device).eval()

    # Predicciones del modelo en todo el conjunto de prueba
    def predict(model):
        _p, _y = [], []
        with torch.no_grad():
            for _x, _t in test_loader:
                _p.append(model(_x.to(device)).argmax(dim=1).cpu())
                _y.append(_t)
        return torch.cat(_p), torch.cat(_y)
    return load_vit, predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Accuracy por clase
    Se evalúan **todos** los checkpoints en el conjunto de prueba completo y se guardan en `results/inspection_<corrida>.json`. Cada clase tiene 1,000 imágenes de prueba, así que la accuracy por clase tiene un error de aproximadamente $\pm 2.5$ puntos (95%): no interpretar diferencias menores.
    """)
    return


@app.cell
def _(RUN_DIR, base, ckpt, curves, json, load_vit, os, predict, torch):
    # evals[(tag, ronda)] = {pct, acc, per_class}
    evals, preds = {}, {}

    def _per_class(p, y):
        return [((p == y) & (y == c)).sum().item() / (y == c).sum().item() for c in range(10)]

    _dense = load_vit(base["dense"])
    _p, _y = predict(_dense)
    preds["dense"] = _p
    evals[("dense", 0)] = {"pct": 100.0, "acc": (_p == _y).float().mean().item(), "per_class": _per_class(_p, _y)}
    for _tag, _rows in curves.items():
        for _r, _pct, _, _path in _rows:
            _p, _y = predict(load_vit(ckpt(_path)["model"]))
            preds[(_tag, _r)] = _p
            evals[(_tag, _r)] = {"pct": _pct, "acc": (_p == _y).float().mean().item(), "per_class": _per_class(_p, _y)}
    labels = _y

    os.makedirs("../results", exist_ok=True)
    with open(f"../results/inspection_{os.path.basename(RUN_DIR.rstrip('/'))}.json", "w") as _f:
        json.dump({f"{t}_r{r}": v for (t, r), v in evals.items()}, _f)
    print(f"{len(evals)} checkpoints evaluados; denso: acc = {evals[('dense', 0)]['acc']:.4f}")
    return evals, labels, preds


@app.cell
def _(at_target, classes, curves, evals, np, plt):
    _sel = [("dense", evals[("dense", 0)])] + [(t, evals[(t, at_target(t)[0])]) for t in curves if t in ("ticket", "late", "random")]
    _x = np.arange(10)
    _w = 0.8 / len(_sel)
    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for _i, (_name, _e) in enumerate(_sel):
        _lab = _name if _name == "dense" else f"{_name} ({_e['pct']:.1f}%)"
        _ax1.bar(_x + _i * _w, _e["per_class"], _w, label=_lab)
        _ax2.bar(_x + _i * _w, np.array(_e["per_class"]) - np.array(_sel[0][1]["per_class"]), _w, label=_lab)
    _ax1.set_ylabel("Accuracy por clase")
    _ax1.legend(fontsize=8)
    _ax2.axhline(0, color="k", lw=0.5)
    _ax2.axhspan(-0.025, 0.025, color="gray", alpha=0.2, label="~ruido (±2.5 pts)")
    _ax2.set_ylabel("Diferencia vs denso")
    _ax2.set_xticks(_x + 0.4)
    _ax2.set_xticklabels(classes, rotation=45)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(at_target, classes, labels, plt, preds, torch):
    # Matriz de confusión del denso y del ticket más cercano al porcentaje objetivo
    def confusion(p):
        _c = torch.zeros(10, 10, dtype=torch.long)
        for _t, _q in zip(labels, p):
            _c[_t, _q] += 1
        return _c

    _t = ("ticket", at_target("ticket")[0])
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for _ax, _name, _p in zip(_axes, ["denso", f"ticket ({_t[1]})"], [preds["dense"], preds[_t]]):
        _ax.imshow(confusion(_p), cmap="Blues")
        _ax.set_title(_name)
        _ax.set_xticks(range(10))
        _ax.set_xticklabels(classes, rotation=90, fontsize=7)
        _ax.set_yticks(range(10))
        _ax.set_yticklabels(classes, fontsize=7)
        _ax.set_xlabel("Predicha")
        _ax.set_ylabel("Real")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mapas de atención
    Atención del token `[CLS]` hacia los 64 parches (8x8) en el **último bloque**, promediada sobre las cabezas, para las 10 imágenes fijas. Cada fila es un modelo distinto al mismo porcentaje de pesos. El `random` tiene la misma máscara que el ticket pero otra inicialización, así que si se ve distinto es por la inicialización.
    """)
    return


@app.cell
def _(at_target, base, ckpt, curves, device, fixed_x, load_vit, mean, plt, std, torch):
    _rows = [("denso", load_vit(base["dense"]))]
    for _tag in ("ticket", "late", "random"):
        if _tag in curves:
            _c = at_target(_tag)
            _rows.append((f"{_tag} ({_c[1]:.1f}%)", load_vit(ckpt(_c[3])["model"])))

    _fig, _axes = plt.subplots(len(_rows) + 1, 10, figsize=(14, 1.5 * (len(_rows) + 1)))
    for _j in range(10):
        _axes[0, _j].imshow((fixed_x[_j] * std + mean).permute(1, 2, 0).clamp(0, 1))
    _axes[0, 0].set_ylabel("imagen", fontsize=8)
    for _i, (_name, _m) in enumerate(_rows, start=1):
        with torch.no_grad():
            # [B, heads, N, N] del último bloque -> fila del CLS hacia los parches, promedio de cabezas
            _attn = _m.all_attention(fixed_x.to(device))[-1][:, :, 0, 1:].mean(dim=1).cpu()
        for _j in range(10):
            _axes[_i, _j].imshow(_attn[_j].reshape(8, 8), cmap="inferno")
        _axes[_i, 0].set_ylabel(_name, fontsize=8)
    for _a in _axes.flat:
        _a.set_xticks([])
        _a.set_yticks([])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Entropía de la atención vs porcentaje de pesos
    Entropía de la fila del `[CLS]` (65 llaves), promediada sobre 512 imágenes, cabezas y bloques. Entropía máxima ($\ln 65 \approx 4.17$) = atención uniforme; cercana a 0 = una sola llave. Si la entropía cae o sube al podar, las cabezas están cambiando de comportamiento.
    """)
    return


@app.cell
def _(base, ckpt, curves, device, load_vit, plt, sample_x, torch):
    def mean_entropy(model):
        with torch.no_grad():
            _vals = []
            for _a in model.all_attention(sample_x.to(device)):
                _row = _a[:, :, 0, :]
                _vals.append(-(_row * torch.log(_row + 1e-12)).sum(-1).mean().item())
        return sum(_vals) / len(_vals)

    _fig, _ax = plt.subplots()
    _ax.axhline(mean_entropy(load_vit(base["dense"])), color="k", ls="--", label="denso")
    for _tag, _rows in curves.items():
        _ax.plot([c[1] for c in _rows], [mean_entropy(load_vit(ckpt(c[3])["model"])) for c in _rows], marker="o", label=_tag)
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Weights remaining (%)")
    _ax.set_ylabel("Entropía media de la atención del CLS")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Gradientes por capa
    Norma del gradiente de la pérdida respecto a los pesos que **sobreviven**, por tipo de capa (promediada sobre los bloques), en 512 imágenes de prueba. Dos momentos al porcentaje objetivo:

    - **Paso 0:** antes de entrenar, con la máscara aplicada a los pesos iniciales (`ticket`), a los de rewind (`late`) o a una inicialización aleatoria nueva (`random`, guardada en el checkpoint). Muestra si la inicialización del ticket arranca con gradientes distintos a una aleatoria con la misma estructura.
    - **Final:** el modelo ya entrenado. Aquí los gradientes son pequeños, sirve más para comparar entre curvas que en valor absoluto.
    """)
    return


@app.cell
def _(at_target, base, ckpt, curves, device, load_vit, nn, np, plt, sample_x, sample_y, torch):
    import re

    # "blocks.3.mlp.0.weight" -> "mlp.0"
    def layer_type(name):
        return re.sub(r"^blocks\.\d+\.", "", name).removesuffix(".weight")

    def grad_by_type(model, masks):
        model.zero_grad()
        nn.CrossEntropyLoss()(model(sample_x.to(device)), sample_y.to(device)).backward()
        _sums = {}
        for _n, _p in model.named_parameters():
            if _n in masks:
                _g = (_p.grad * masks[_n].to(device)).norm().item()
                _sums.setdefault(layer_type(_n), []).append(_g)
        return {k: float(np.mean(v)) for k, v in _sums.items()}

    def with_state(state, masks):
        _m = load_vit(state)
        # Aplicar la máscara al estado (el rewind / init no la traen aplicada)
        with torch.no_grad():
            for _n, _p in _m.named_parameters():
                if _n in masks:
                    _p.mul_(masks[_n].to(device))
        return _m

    _t = at_target("ticket")
    _masks = {"ticket": ckpt(_t[3])["masks"]}
    _start, _end = {}, {}
    _start["ticket"] = grad_by_type(with_state(base["init"], _masks["ticket"]), _masks["ticket"])
    _end["ticket"] = grad_by_type(load_vit(ckpt(_t[3])["model"]), _masks["ticket"])
    if "late" in curves:
        _l = at_target("late")
        _masks["late"] = ckpt(_l[3])["masks"]
        if base["rewind"] is not None:
            _start["late"] = grad_by_type(with_state(base["rewind"], _masks["late"]), _masks["late"])
        _end["late"] = grad_by_type(load_vit(ckpt(_l[3])["model"]), _masks["late"])
    if "random" in curves:
        _r = at_target("random")
        _ck = ckpt(_r[3])
        _start["random"] = grad_by_type(load_vit(_ck["init_model"]), _ck["masks"])
        _end["random"] = grad_by_type(load_vit(_ck["model"]), _ck["masks"])

    _types = list(_start["ticket"].keys())
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for _ax, _data, _title in zip(_axes, [_start, _end], ["Paso 0 (pesos iniciales con máscara)", "Final (modelo entrenado)"]):
        _w = 0.8 / len(_data)
        for _i, (_name, _d) in enumerate(_data.items()):
            _ax.bar(np.arange(len(_types)) + _i * _w, [_d[t] for t in _types], _w, label=_name)
        _ax.set_xticks(np.arange(len(_types)) + 0.4)
        _ax.set_xticklabels(_types, rotation=45)
        _ax.set_yscale("log")
        _ax.set_title(_title)
        _ax.set_ylabel("Norma del gradiente (pesos vivos)")
        _ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
