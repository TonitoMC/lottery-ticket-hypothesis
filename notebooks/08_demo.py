import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo: tickets de la Lottery Ticket Hypothesis en acción
    **Proyecto Final - Deep Learning.** Aquí se *usan* los modelos guardados en `checkpoints/`: se elige una red, un nivel de poda y una imagen, y se comparan las predicciones de

    - **denso:** la red completa entrenada,
    - **ticket:** la subred podada, reentrenada desde la inicialización original (el winning ticket),
    - **aleatorio:** la **misma** subred (misma máscara) pero con pesos nuevos al azar (el control),
    - **rewinding:** solo en el ViT, la subred reentrenada desde la época 5.

    Qué probar en el video: (1) con la CNN global y 10.7% de los pesos, el ticket sigue acertando casi igual que el denso y el aleatorio falla más; (2) con la CNN por capa y 10.7% el ticket es *peor* que el aleatorio (el colapso); (3) con el ViT y 3-5% de los pesos, la diferencia ticket / rewinding / aleatorio es grande; (4) subir tu propia imagen.

    *Para usarlo en vivo:* `cd notebooks && uv run marimo run 08_demo.py` (o `marimo edit`). El HTML exportado muestra una configuración fija.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import io
    import json
    import os
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt
    from PIL import Image

    import lth

    device = "cpu"  # basta para una imagen a la vez; así corre mientras la GPU entrena otra cosa
    return F, Image, datasets, device, io, json, lth, mo, nn, np, os, plt, torch, v2


@app.cell
def _(F, nn, torch):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.linear = nn.Sequential(nn.Flatten(), nn.Linear(8192, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

        def forward(self, x):
            return self.linear(self.conv(x))

    class Block(nn.Module):
        def __init__(self, dim, heads, mlp_dim):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

        def attention(self, x):
            B, N, D = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            return torch.softmax(q @ k.transpose(-2, -1) / (D // self.heads) ** 0.5, dim=-1)

        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
            return x + self.mlp(self.norm2(x))

    class ViT(nn.Module):
        def __init__(self, dim=192, depth=6, heads=3, mlp_dim=384, patch=4):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, (32 // patch) ** 2 + 1, dim) * 0.02)
            self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 10)

        def embed(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            return torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos

        def forward(self, x):
            return self.head(self.norm(self.blocks(self.embed(x)))[:, 0])

        # Atención del [CLS] en el último bloque, promedio de cabezas: [B, 64]
        def cls_attention(self, x):
            x = self.embed(x)
            for blk in self.blocks[:-1]:
                x = blk(x)
            return self.blocks[-1].attention(x)[:, :, 0, 1:].mean(1)

    MODELS = {
        "CNN, poda por capa (02)": dict(dir="../checkpoints/02_cnn", res="cnn_cifar10", cls=CNN, norm=False),
        "CNN, poda global (02b)": dict(dir="../checkpoints/02b_cnn_global", res="cnn_global_cifar10", cls=CNN, norm=False),
        "ViT largo, poda global (05)": dict(dir="../checkpoints/05_vit_longer", res="vit_longer", cls=ViT, norm=True),
    }
    return MODELS, ViT


@app.cell
def _(datasets, torch, v2):
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    CLASSES = ["avión", "auto", "pájaro", "gato", "venado", "perro", "rana", "caballo", "barco", "camión"]
    MEAN, STD = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1), torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return CLASSES, MEAN, STD, test_set


@app.cell
def _(MODELS, lth, mo):
    import functools

    @functools.lru_cache(maxsize=None)
    def list_ckpts(name):
        # {tag: [(ronda, % pesos, acc, ruta)]}; el checkpoint se lee solo para su metadato
        out = {}
        for tag, r, pct, acc, path in lth.list_checkpoints(MODELS[name]["dir"]):
            out.setdefault(tag, []).append((r, pct, acc, path))
        return {t: sorted(v) for t, v in out.items()}

    model_pick = mo.ui.dropdown(list(MODELS), value="CNN, poda global (02b)", label="Modelo")
    return list_ckpts, model_pick


@app.cell
def _(list_ckpts, mo, model_pick):
    # Solo hay checkpoints de algunos niveles (los más significativos); se ofrecen los disponibles
    _rows = list_ckpts(model_pick.value)["ticket"]
    _opts = {f"{r[1]:.1f}% de los pesos (ronda {r[0]})": r[0] for r in _rows}
    round_pick = mo.ui.dropdown(_opts, value=list(_opts)[-1], label="Nivel de poda")
    return (round_pick,)


@app.cell
def _(mo, test_set):
    img_pick = mo.ui.slider(0, len(test_set) - 1, value=7, label="Imagen de prueba (CIFAR-10)", show_value=True)
    upload = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"], label="O sube tu propia imagen (se reduce a 32x32)")
    mo.vstack([mo.hstack([img_pick, upload], justify="start", gap=2)])
    return img_pick, upload


@app.cell
def _(Image, MEAN, STD, CLASSES, img_pick, io, np, test_set, torch, upload):
    if upload.value:
        _im = Image.open(io.BytesIO(upload.value[0].contents)).convert("RGB")
        _w, _h = _im.size
        _s = min(_w, _h)
        _im = _im.crop(((_w - _s) // 2, (_h - _s) // 2, (_w + _s) // 2, (_h + _s) // 2)).resize((32, 32), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(_im).copy()).permute(2, 0, 1).float() / 255
        truth = None
        source = "imagen subida"
    else:
        x, _y = test_set[img_pick.value]
        truth = CLASSES[_y]
        source = f"CIFAR-10 de prueba #{img_pick.value}"
    return source, truth, x


@app.cell
def _(MEAN, MODELS, STD, ViT, device, json, list_ckpts, lth, model_pick, round_pick, torch, x):
    spec = MODELS[model_pick.value]
    _ck = list_ckpts(model_pick.value)
    _base = torch.load(f"{spec['dir']}/base.pt", map_location="cpu")
    _inp = ((x - MEAN) / STD if spec["norm"] else x).unsqueeze(0).to(device)

    def _load(state):
        _m = spec["cls"]()
        _m.load_state_dict(state)
        return _m.to(device).eval()

    # (nombre, modelo, % de pesos, acc de prueba completa)
    _dense_acc = json.load(open(f"../results/{spec['res']}.json"))["ticket"][0]["acc"]
    variants = [("denso", _load(_base["dense"]), 100.0, _dense_acc)]
    for _tag, _name in [("ticket", "ticket"), ("late", "rewinding"), ("random", "aleatorio")]:
        _row = next((c for c in _ck.get(_tag, []) if c[0] == round_pick.value), None)
        if _row:
            variants.append((_name, _load(lth.load_checkpoint(_row[3])["model"]), _row[1], _row[2]))
    with torch.no_grad():
        outputs = {n: torch.softmax(m(_inp), dim=1)[0] for n, m, _, _ in variants}
    attn = None
    if isinstance(variants[0][1], ViT):
        with torch.no_grad():
            attn = {n: m.cls_attention(_inp)[0].reshape(8, 8) for n, m, _, _ in variants}
    inp = _inp
    return attn, outputs, variants


@app.cell
def _(CLASSES, attn, mo, outputs, plt, source, truth, variants, x):
    _n = len(variants)
    _fig, _axes = plt.subplots(2 if attn else 1, _n + 1, figsize=(3.0 * (_n + 1), 6.2 if attn else 3.4), squeeze=False)
    _axes[0][0].imshow(x.permute(1, 2, 0).clamp(0, 1), interpolation="nearest")
    _axes[0][0].set_title(f"{source}\nreal: {truth or '—'}", fontsize=9)
    _axes[0][0].axis("off")
    for _i, (_name, _m, _pct, _acc) in enumerate(variants, start=1):
        _p = outputs[_name]
        _top = _p.topk(3)
        _ax = _axes[0][_i]
        _ax.barh([CLASSES[k] for k in _top.indices.tolist()][::-1], _top.values.tolist()[::-1],
                 color=["#2ca02c" if (truth and CLASSES[k] == truth) else "#1f77b4" for k in _top.indices.tolist()][::-1])
        _ax.set_xlim(0, 1)
        _ax.set_title(f"{_name} ({_pct:.1f}% pesos)\nacc total: {_acc:.1%}", fontsize=9)
        _ax.tick_params(labelsize=8)
    if attn:
        _axes[1][0].axis("off")
        for _i, (_name, *_rest) in enumerate(variants, start=1):
            _axes[1][_i].imshow(x.permute(1, 2, 0).clamp(0, 1), interpolation="nearest")
            _axes[1][_i].imshow(attn[_name], extent=(0, 32, 32, 0), cmap="inferno", alpha=0.6, interpolation="bilinear")
            _axes[1][_i].set_title(f"atención [CLS]: {_name}", fontsize=8)
            _axes[1][_i].axis("off")
    _fig.tight_layout()
    _guess = ", ".join(f"**{n}** → {CLASSES[int(p.argmax())]} ({p.max():.0%})" for n, p in outputs.items())
    mo.vstack([
        _fig,
        mo.md(f"**Predicciones:** {_guess}."),
        mo.md("*Cómo leerlo:* arriba, las 3 clases con más probabilidad de cada modelo (verde = la clase real, si se conoce). 'acc total' es la accuracy de ese checkpoint en las 10,000 imágenes de prueba, así que es la medida confiable; una sola imagen no demuestra nada, sirve para ilustrar. "
              "Abajo (solo ViT), cuánto mira el token [CLS] cada parche: más brillante = más atención. Si subes una imagen que no es de CIFAR-10 (fotos grandes, otras clases), los modelos solo conocen las 10 clases y una imagen de 32x32, así que pueden fallar."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curvas completas
    Accuracy de prueba en todos los niveles de poda del modelo elegido, con la ronda seleccionada marcada. Esta es la evidencia del experimento; la sección de arriba es solo para ilustrarla.
    """)
    return


@app.cell
def _(MODELS, json, list_ckpts, model_pick, plt, round_pick):
    _res = json.load(open(f"../results/{MODELS[model_pick.value]['res']}.json"))
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _key, _lab in [("ticket", "ticket"), ("ticket_late", "rewinding"), ("random", "aleatorio")]:
        if _key in _res:
            _ax.plot([c["weights_left"] * 100 for c in _res[_key]], [c["acc"] for c in _res[_key]], marker="o", markersize=4, label=_lab)
    _sel = next(c[1] for c in list_ckpts(model_pick.value)["ticket"] if c[0] == round_pick.value)
    _ax.axvline(_sel, color="gray", ls=":")
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Pesos restantes (%)")
    _ax.set_ylabel("Accuracy de prueba")
    _ax.grid(alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
