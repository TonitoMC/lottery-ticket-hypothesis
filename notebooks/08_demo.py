import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Demo LTH")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo: tickets de la Lottery Ticket Hypothesis en acción
    Se eligen una red, un nivel de poda y una imagen, y se comparan cuatro modelos con **la misma estructura**:
    **denso** (red completa), **ticket** (subred reentrenada desde sus pesos iniciales), **rewinding** (solo ViT, desde la época 5) y **aleatorio** (la misma máscara con pesos nuevos: el control).
    *Uso en vivo:* `cd notebooks && uv run marimo run 08_demo.py`.
    """)
    return


@app.cell(hide_code=True)
def _(attn_pick, img_pick, level_pick, mo, mode_pick, model_pick, upload):
    mo.vstack([
        mo.hstack([model_pick, level_pick, mode_pick], justify="start", gap=2),
        mo.hstack([img_pick, upload, attn_pick], justify="start", gap=2, align="end"),
    ])
    return


@app.cell(hide_code=True)
def _(card_grid, mo):
    mo.vstack([card_grid])
    return


@app.cell(hide_code=True)
def _(attn_fig, mo):
    attn_fig if attn_fig is not None else mo.md("")
    return


@app.cell(hide_code=True)
def _(curves_fig, mo):
    mo.vstack([
        mo.md("### Accuracy de todo el experimento\nLa línea gris marca el nivel elegido. **Esta es la evidencia**; la imagen de arriba solo ilustra."),
        curves_fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    import base64
    import functools
    import io
    import json
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

    device = "cpu"  # una imagen a la vez: la CPU basta
    return (
        F,
        Image,
        base64,
        datasets,
        device,
        functools,
        io,
        json,
        lth,
        mo,
        nn,
        np,
        plt,
        torch,
        v2,
    )


@app.cell(hide_code=True)
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
        "ViT largo, poda global": dict(dir="../checkpoints/05_vit_longer", res="vit_longer", cls=ViT, norm=True),
        "CNN, poda global": dict(dir="../checkpoints/02b_cnn_global", res="cnn_global_cifar10", cls=CNN, norm=False),
        "CNN, poda por capa": dict(dir="../checkpoints/02_cnn", res="cnn_cifar10", cls=CNN, norm=False),
    }
    return MODELS, ViT


@app.cell(hide_code=True)
def _(datasets, torch, v2):
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    CLASSES = ["avión", "auto", "pájaro", "gato", "venado", "perro", "rana", "caballo", "barco", "camión"]
    MEAN, STD = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1), torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    N_POOL = 1000  # las primeras 1,000 imágenes de prueba sirven para buscar casos interesantes
    X_pool = torch.stack([test_set[i][0] for i in range(N_POOL)])
    Y_pool = torch.tensor(test_set.targets[:N_POOL])
    return CLASSES, MEAN, STD, X_pool, Y_pool, test_set


@app.cell(hide_code=True)
def _(MEAN, MODELS, STD, X_pool, Y_pool, device, functools, json, lth, torch):
    @functools.lru_cache(maxsize=None)
    def list_ckpts(name):
        # {tag: [(ronda, % de pesos, acc, ruta)]}; el checkpoint solo se lee por su metadato
        out = {}
        for tag, r, pct, acc, path in lth.list_checkpoints(MODELS[name]["dir"]):
            out.setdefault(tag, []).append((r, pct, acc, path))
        return {t: sorted(v) for t, v in out.items()}

    @functools.lru_cache(maxsize=None)
    def get_variants(name, rnd):
        """[(nombre, modelo, % de pesos, acc de prueba completa)] para denso, ticket, rewinding (si hay) y aleatorio."""
        spec, ck = MODELS[name], list_ckpts(name)
        base = torch.load(f"{spec['dir']}/base.pt", map_location="cpu")

        def _load(state):
            m = spec["cls"]()
            m.load_state_dict(state)
            return m.to(device).eval()

        dense_acc = json.load(open(f"../results/{spec['res']}.json"))["ticket"][0]["acc"]
        out = [("denso", _load(base["dense"]), 100.0, dense_acc)]
        for tag, label in [("ticket", "ticket"), ("late", "rewinding"), ("random", "aleatorio")]:
            row = next((c for c in ck.get(tag, []) if c[0] == rnd), None)
            if row:
                out.append((label, _load(lth.load_checkpoint(row[3])["model"]), row[1], row[2]))
        return tuple(out)

    def prep(name, x):
        return ((x - MEAN) / STD) if MODELS[name]["norm"] else x

    @functools.lru_cache(maxsize=None)
    def pool_preds(name, rnd):
        """Predicciones de cada variante sobre las 1,000 imágenes del grupo."""
        with torch.no_grad():
            Xp = prep(name, X_pool)
            return {n: torch.cat([m(Xp[i:i + 250]).argmax(1) for i in range(0, len(Xp), 250)]) for n, m, _, _ in get_variants(name, rnd)}

    def pool_indices(name, rnd, mode):
        p = pool_preds(name, rnd)
        if mode == "win":
            keep = (p["ticket"] == Y_pool) & (p["aleatorio"] != Y_pool)
        elif mode == "lose":
            keep = p["ticket"] != Y_pool
        else:
            keep = torch.ones_like(Y_pool, dtype=torch.bool)
        idx = keep.nonzero().flatten().tolist()
        return idx if idx else list(range(len(Y_pool)))

    return get_variants, list_ckpts, pool_indices, prep


@app.cell(hide_code=True)
def _(MODELS, mo):
    model_pick = mo.ui.dropdown(list(MODELS), value="ViT largo, poda global", label="Modelo")
    mode_pick = mo.ui.dropdown(
        {"Ticket acierta y el control falla": "win", "El ticket falla": "lose", "Todas": "all"},
        value="Ticket acierta y el control falla",
        label="Qué imágenes ver",
    )
    attn_pick = mo.ui.checkbox(value=True, label="Mostrar atención (solo ViT)")
    upload = mo.ui.file(kind="button", filetypes=[".png", ".jpg", ".jpeg"], label="Subir mi imagen")
    return attn_pick, mode_pick, model_pick, upload


@app.cell(hide_code=True)
def _(list_ckpts, mo, model_pick):
    # solo hay checkpoints de algunos niveles (los más informativos)
    _rows = list_ckpts(model_pick.value)["ticket"]
    _opts = {f"{r[1]:.1f}% de los pesos": r[0] for r in _rows}
    level_pick = mo.ui.dropdown(_opts, value=list(_opts)[-1], label="Nivel de poda")
    return (level_pick,)


@app.cell(hide_code=True)
def _(level_pick, mo, mode_pick, model_pick, pool_indices):
    pool = pool_indices(model_pick.value, level_pick.value, mode_pick.value)
    img_pick = mo.ui.slider(0, max(len(pool) - 1, 0), value=0, label=f"Imagen ({len(pool)} disponibles)", show_value=False, full_width=False)
    return img_pick, pool


@app.cell(hide_code=True)
def _(Image, img_pick, io, np, pool, test_set, torch, upload):
    if upload.value:
        _im = Image.open(io.BytesIO(upload.value[0].contents)).convert("RGB")
        _w, _h = _im.size
        _s = min(_w, _h)
        _im = _im.crop(((_w - _s) // 2, (_h - _s) // 2, (_w + _s) // 2, (_h + _s) // 2)).resize((32, 32), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(_im).copy()).permute(2, 0, 1).float() / 255
        truth = None
        source = "tu imagen (reducida a 32×32)"
    else:
        _idx = pool[min(img_pick.value, len(pool) - 1)]
        x, _y = test_set[_idx]
        truth = int(_y)
        source = f"CIFAR-10 de prueba #{_idx}"
    return source, truth, x


@app.cell(hide_code=True)
def _(
    CLASSES,
    Image,
    base64,
    get_variants,
    io,
    level_pick,
    mo,
    model_pick,
    prep,
    source,
    torch,
    truth,
    x,
):
    variants = get_variants(model_pick.value, level_pick.value)
    _inp = prep(model_pick.value, x).unsqueeze(0)
    with torch.no_grad():
        probs = {n: torch.softmax(m(_inp), dim=1)[0] for n, m, _, _ in variants}

    def _png(t):
        im = Image.fromarray((t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")).resize((224, 224), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _card(name, pct, acc):
        p = probs[name]
        top = p.topk(3)
        pred = int(top.indices[0])
        ok = None if truth is None else pred == truth
        color = "#2a9d6f" if ok else ("#d9534f" if ok is False else "#8a8fa8")
        mark = "" if ok is None else (" ✓" if ok else " ✗")
        bars = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin-top:3px">'
            f'<span style="width:62px;opacity:.75">{CLASSES[int(i)]}</span>'
            f'<div style="flex:1;background:rgba(128,128,128,.2);border-radius:4px;height:8px"><div style="width:{float(v) * 100:.0f}%;background:{color if k == 0 else "#8a8fa8"};height:8px;border-radius:4px"></div></div>'
            f'<span style="width:34px;text-align:right;opacity:.75">{float(v):.0%}</span></div>'
            for k, (v, i) in enumerate(zip(top.values, top.indices))
        )
        return (
            f'<div style="border:2px solid {color};border-radius:12px;padding:12px 14px;min-width:190px;flex:1">'
            f'<div style="font-weight:700;font-size:15px">{name}</div>'
            f'<div style="font-size:12px;opacity:.7;margin-bottom:6px">{pct:.1f}% de los pesos · acc. total {acc:.1%}</div>'
            f'<div style="font-size:22px;font-weight:700;color:{color}">{CLASSES[pred]}{mark}</div>'
            f'<div style="font-size:12px;opacity:.7;margin-bottom:4px">confianza {float(top.values[0]):.0%}</div>{bars}</div>'
        )

    _truth_txt = f"real: <b>{CLASSES[truth]}</b>" if truth is not None else "sin etiqueta real"
    _left = (
        f'<div style="min-width:240px"><img src="data:image/png;base64,{_png(x)}" style="width:224px;height:224px;border-radius:10px;image-rendering:pixelated">'
        f'<div style="font-size:13px;margin-top:6px">{source}<br>{_truth_txt}</div></div>'
    )
    _cards = "".join(_card(n, pct, acc) for n, _, pct, acc in variants)
    card_grid = mo.Html(
        f'<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap">{_left}'
        f'<div style="display:flex;gap:12px;flex:1;flex-wrap:wrap">{_cards}</div></div>'
        f'<div style="font-size:12px;opacity:.65;margin-top:10px">Verde = acierta, rojo = falla. Una sola imagen no demuestra nada: lo confiable es la '
        f'accuracy total (10,000 imágenes de prueba) de cada tarjeta.</div>'
    )
    return card_grid, variants


@app.cell(hide_code=True)
def _(ViT, attn_pick, mo, model_pick, plt, prep, torch, variants, x):
    attn_fig = None
    if attn_pick.value and isinstance(variants[0][1], ViT):
        _inp = prep(model_pick.value, x).unsqueeze(0)
        _fig, _axes = plt.subplots(1, len(variants), figsize=(2.6 * len(variants), 2.9), squeeze=False)
        for _ax, (_n, _m, _pct, _acc) in zip(_axes[0], variants):
            with torch.no_grad():
                _a = _m.cls_attention(_inp)[0].reshape(8, 8)
            _ax.imshow(x.permute(1, 2, 0).clamp(0, 1), interpolation="nearest")
            _ax.imshow(_a, extent=(0, 32, 32, 0), cmap="inferno", alpha=0.6, interpolation="bilinear")
            _ax.set_title(_n, fontsize=10)
            _ax.axis("off")
        _fig.tight_layout()
        attn_fig = mo.vstack([
            mo.md("### Atención del token [CLS]\n*Descriptiva, no una explicación:* en el informe se midió que la atención cruda **no** predice qué partes de la imagen usa el modelo (correlación ≈ 0 con la oclusión). Sí se ve que el control aleatorio reparte la atención distinto."),
            _fig,
        ])
    return (attn_fig,)


@app.cell(hide_code=True)
def _(MODELS, json, level_pick, list_ckpts, model_pick, plt):
    _res = json.load(open(f"../results/{MODELS[model_pick.value]['res']}.json"))
    _fig, _ax = plt.subplots(figsize=(7.5, 3.4))
    for _key, _lab, _c in [("ticket", "ticket", "#1f77b4"), ("ticket_late", "rewinding", "#2a9d6f"), ("random", "aleatorio", "#ff7f0e")]:
        if _key in _res:
            _ax.plot([c["weights_left"] * 100 for c in _res[_key]], [c["acc"] for c in _res[_key]], marker="o", markersize=4, label=_lab, color=_c, ls="--" if _key == "random" else "-")
    _sel = next(c[1] for c in list_ckpts(model_pick.value)["ticket"] if c[0] == level_pick.value)
    _ax.axvline(_sel, color="gray", ls=":")
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Pesos restantes (%)")
    _ax.set_ylabel("Accuracy de prueba")
    _ax.grid(alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
    curves_fig = _fig
    return (curves_fig,)


if __name__ == "__main__":
    app.run()
