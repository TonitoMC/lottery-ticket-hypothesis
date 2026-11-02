import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
        Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

        ## 07 - Resultados e inspección de los tickets
    Este notebook reúne la parte **objetiva** del proyecto, sin discusión: primero los resultados de todas las corridas (curvas, resumen, arquitecturas, comparación de máscaras de la CNN) y luego la inspección de los checkpoints. La interpretación está en `report/informe.tex`.

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
    return (
        DataLoader,
        F,
        RUN_DIR,
        TARGET_PCT,
        datasets,
        device,
        json,
        lth,
        mo,
        nn,
        np,
        os,
        plt,
        torch,
        v2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resultados de todas las corridas
    Esta parte no necesita checkpoints: lee `results/*.json` y, para la comparación de máscaras de la CNN, `checkpoints/02_cnn` y `checkpoints/02b_cnn_global`. Es solo descripción de los datos; la interpretación (hipótesis, discusión, conclusiones) está en `report/informe.tex`. Todas las corridas son de una sola semilla: diferencias de ~1 punto o menos son ruido.

    Tres curvas por arquitectura: **ticket** (los pesos sobrevivientes regresan a su valor inicial), **ticket con rewinding** (regresan a su valor tras $k$ épocas de entrenamiento denso, solo ResNet y ViT) y **reinicialización aleatoria** (la misma máscara con pesos nuevos, el control).
    """)
    return


@app.cell
def _(json, os):
    EXPERIMENTS = {
        "mlp_mnist": "MLP / MNIST",
        "cnn_cifar10": "CNN poda por capa / CIFAR-10",
        "cnn_global_cifar10": "CNN poda global / CIFAR-10",
        "resnet18_cifar10": "ResNet-20 / CIFAR-10",
        "vit_cifar10": "ViT (20 ép.) / CIFAR-10",
        "vit_longer": "ViT largo (100 ép.) / CIFAR-10",
    }
    runs = {}
    for _file, _name in EXPERIMENTS.items():
        _path = os.path.join("..", "results", f"{_file}.json")
        if os.path.exists(_path):
            with open(_path) as _f:
                runs[_name] = json.load(_f)
    _dense = os.path.join("..", "results", "vit_longer_dense.json")
    vit_dense = json.load(open(_dense)) if os.path.exists(_dense) else None
    print("corridas cargadas:", ", ".join(runs))
    return runs, vit_dense


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arquitecturas e IMP
    Las cuatro redes con la forma de los tensores y los pesos podables por capa (verde = capas que se podan), y el ciclo de poda iterativa por magnitud (IMP): entrenar con la máscara, quitar el 20% (35% en el ViT) de los pesos vivos de menor magnitud, regresar los sobrevivientes al punto de referencia y volver a entrenar.
    """)
    return


@app.cell
def _(plt):
    from matplotlib.patches import FancyBboxPatch

    def _row(ax, y, items, title, colors=None):
        # items: [(texto, ancho)]; se dibujan cajas con flechas entre ellas
        ax.text(0.0, y + 0.62, title, fontsize=10, fontweight="bold", va="bottom")
        x = 0.0
        for i, (txt, w) in enumerate(items):
            c = (colors or {}).get(i, "#dbe9f6")
            ax.add_patch(FancyBboxPatch((x, y), w, 0.55, boxstyle="round,pad=0.02", fc=c, ec="#33506b", lw=0.8))
            ax.text(x + w / 2, y + 0.275, txt, ha="center", va="center", fontsize=7)
            if i < len(items) - 1:
                ax.annotate("", xy=(x + w + 0.18, y + 0.275), xytext=(x + w + 0.02, y + 0.275),
                            arrowprops=dict(arrowstyle="->", color="#33506b", lw=0.8))
            x += w + 0.2
        return x

    _fig, _ax = plt.subplots(figsize=(14, 7.6))
    _ax.set_xlim(-0.2, 15.2)
    _ax.set_ylim(-0.2, 10.4)
    _ax.axis("off")
    _prune, _norm, _act = "#cfe8cf", "#eeeeee", "#fbe3c8"
    _row(_ax, 9.0, [("Imagen\n(784)", 1.0), ("Linear\n784→300\n235k", 1.2), ("ReLU", 0.7), ("Linear\n300→100\n30k", 1.2), ("ReLU", 0.7), ("Linear\n100→10\n1k", 1.1), ("10 clases", 1.0)],
         "MLP (LeNet-300-100) · MNIST · 266k pesos podables", {1: _prune, 3: _prune, 5: _prune, 2: _act, 4: _act})
    _row(_ax, 7.2, [("Imagen\n(3,32,32)", 1.1), ("Conv 3→64\n1.7k", 1.0), ("Conv 64→64\n37k", 1.0), ("MaxPool\n(64,16,16)", 0.95), ("Conv 64→128\n74k", 1.05), ("Conv 128→128\n147k", 1.1), ("MaxPool\n(128,8,8)", 0.95), ("Linear\n8192→256\n2.1M", 1.05), ("Linear\n256→256\n66k", 1.0), ("Linear\n256→10", 0.85)],
         "CNN (Conv-4) · CIFAR-10 · ~2.4M pesos podables (86% están en la primera Linear)", {1: _prune, 2: _prune, 4: _prune, 5: _prune, 7: _prune, 8: _prune, 9: _prune, 3: _norm, 6: _norm})
    _row(_ax, 5.4, [("Imagen\n(3,32,32)", 1.1), ("Conv 3→16\n+BN+ReLU", 1.1), ("3 bloques\n16 canales\n(32×32)", 1.25), ("3 bloques\n32 canales\n(16×16)", 1.2), ("3 bloques\n64 canales\n(8×8)", 1.2), ("AvgPool\n+ Linear\n64→10", 1.1), ("10 clases", 0.9)],
         "ResNet de 20 capas [1] · CIFAR-10 · ~270k pesos podables", {1: _prune, 2: _prune, 3: _prune, 4: _prune, 5: _prune})
    _ax.text(9.6, 5.75, "Bloque residual:  x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ x) → ReLU\n(la suma con x es la conexión residual; con cambio de canales, atajo 1×1)", fontsize=7.5, va="center", color="#444")
    _row(_ax, 3.6, [("Imagen\n(3,32,32)", 1.1), ("Parches 4×4\nConv 3→192\n(64 tokens)", 1.3), ("+ [CLS]\n+ posición\n(65,192)", 1.1), ("× 6 bloques\n(ver abajo)", 1.3), ("LayerNorm", 0.9), ("[CLS] →\nLinear\n192→10", 1.0), ("10 clases", 0.9)],
         "ViT pequeño · CIFAR-10 · ~1.8M pesos podables (6 bloques, dim 192, 3 cabezas)", {1: _prune, 5: _prune, 3: "#e3d8f0"})
    _row(_ax, 1.8, [("x", 0.5), ("LayerNorm", 0.9), ("qkv\n192→576", 0.9), ("Atención\n3 cabezas", 1.0), ("proj\n192→192", 0.9), ("+ x", 0.55), ("LayerNorm", 0.9), ("mlp.0\n192→384", 0.95), ("GELU", 0.6), ("mlp.2\n384→192", 0.95), ("+ x", 0.55)],
         "Un bloque del ViT (se repite 6 veces). Todas las Linear y las LayerNorm entran a la poda global", {2: _prune, 4: _prune, 7: _prune, 9: _prune, 3: "#e3d8f0", 1: _norm, 6: _norm})
    # Ciclo de IMP
    _y = 0.1
    _steps = ["1. Entrenar\n(con la máscara)", "2. Podar 20%\nde los pesos vivos\n(menor |w|)", "3. Regresar los\nsobrevivientes a θ₀\no a θ_k (rewind)", "4. Volver a entrenar\ny medir accuracy"]
    _x = 0.2
    for _i, _t in enumerate(_steps):
        _ax.add_patch(FancyBboxPatch((_x, _y), 2.7, 0.9, boxstyle="round,pad=0.03", fc="#fff2cc", ec="#7f6000", lw=0.8))
        _ax.text(_x + 1.35, _y + 0.45, _t, ha="center", va="center", fontsize=7.5)
        if _i < 3:
            _ax.annotate("", xy=(_x + 3.15, _y + 0.45), xytext=(_x + 2.76, _y + 0.45), arrowprops=dict(arrowstyle="->", lw=1))
        _x += 3.2
    _ax.text(13.1, _y + 0.45, "se repite\n8-15 rondas\n(IMP)", ha="left", va="center", fontsize=8, fontweight="bold")
    _ax.text(0.0, 1.2, "Poda por magnitud iterativa (IMP)", fontsize=10, fontweight="bold")
    _ax.text(15.1, 10.3, "Verde: capas que se podan   Naranja: activación   Gris: sin poda (pooling/norm)   Morado: atención", ha="right", fontsize=7.5, color="#444")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy de prueba contra pesos restantes
    Eje x logarítmico e invertido: la red se hace más pequeña hacia la derecha. Si la curva del ticket se mantiene arriba y la del control cae, la estructura sola no explica el resultado.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(plt, runs):
    _labels = {
        "ticket": "Ticket (época 0)",
        "ticket_late": "Ticket (rewind)",
        "random": "Reinit aleatorio",
    }
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    for _ax, (_name, _run) in zip(_axes.flat, runs.items()):
        for _key, _label in _labels.items():
            if _key in _run:
                _res = _run[_key]
                _ax.plot([r["weights_left"] * 100 for r in _res], [r["acc"] for r in _res], marker="o", markersize=4, label=_label)
        _ax.set_xscale("log")
        _ax.invert_xaxis()
        _ax.set_title(_name, fontsize=10)
        _ax.set_xlabel("Pesos restantes (%)")
        _ax.grid(alpha=0.3)
    for _ax in list(_axes.flat)[len(runs):]:
        _ax.axis("off")
    _axes[0][0].set_ylabel("Accuracy de prueba")
    _axes[1][0].set_ylabel("Accuracy de prueba")
    _axes[0][0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy relativa a la red densa
    A cada curva se le resta la accuracy de su red densa, para comparar arquitecturas con bases distintas. 0 = la subred iguala a la completa. Línea continua: ticket; punteada: control aleatorio.
    """)
    return


@app.cell
def _(plt, runs):
    _fig, _ax = plt.subplots(figsize=(8, 4.8))
    for _i, (_name, _run) in enumerate(runs.items()):
        _color = f"C{_i}"
        _dense = _run["ticket"][0]["acc"]
        _ax.plot([r["weights_left"] * 100 for r in _run["ticket"]], [r["acc"] - _dense for r in _run["ticket"]],
                 marker="o", markersize=4, color=_color, label=f"{_name}: ticket")
        if "random" in _run:
            _ax.plot([r["weights_left"] * 100 for r in _run["random"]], [r["acc"] - _dense for r in _run["random"]],
                     linestyle="--", color=_color, alpha=0.6)
    _ax.axhline(0, color="gray", linewidth=0.8)
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Pesos restantes (%)")
    _ax.set_ylabel("Δ accuracy vs. red densa")
    _ax.set_title("Línea continua: ticket. Línea punteada: reinit aleatorio.", fontsize=9)
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=7)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Resumen
    - **Mín. pesos (ticket / aleatorio):** la red más pequeña que queda a menos de 1 punto de la accuracy densa.
    - **Ventaja del ticket:** diferencia promedio entre el ticket y la reinicialización aleatoria sobre las rondas podadas.
    - **Ganancia del rewinding:** diferencia promedio entre el ticket con rewinding y el clásico.
    """)
    return


@app.cell
def _(mo, runs):
    def _smallest(res, dense):
        _ok = [r["weights_left"] for r in res if r["acc"] >= dense - 0.01]
        return f"{min(_ok):.1%}" if _ok else "ninguna"

    def _mean_gap(a, b):
        _n = min(len(a), len(b)) - 1
        return sum(x["acc"] - y["acc"] for x, y in zip(a[1:], b[1:])) / _n if _n > 0 else float("nan")

    _rows = [
        "| Arquitectura | Acc. densa | Mín. pesos (ticket) | Mín. pesos (aleatorio) | Ventaja del ticket | Ganancia del rewinding |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _name, _run in runs.items():
        _dense = _run["ticket"][0]["acc"]
        _rand = _run.get("random", [])
        _adv = f"{_mean_gap(_run['ticket'], _rand):+.2%}" if len(_rand) > 1 else "—"
        _late = f"{_mean_gap(_run['ticket_late'], _run['ticket']):+.2%}" if len(_run.get("ticket_late", [])) > 1 else "—"
        _rows.append(f"| {_name} | {_dense:.2%} | {_smallest(_run['ticket'], _dense)} | {_smallest(_rand, _dense) if _rand else '—'} | {_adv} | {_late} |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ViT denso de 100 épocas
    Accuracy de entrenamiento (sin aumento de datos) y de prueba por época. Sirve para ver si el baseline contra el que se compara el ticket está sobreajustado o subentrenado.
    """)
    return


@app.cell
def _(plt, vit_dense):
    if vit_dense:
        _h = vit_dense["history"]
        _fig, _ax = plt.subplots(figsize=(6.5, 4))
        _ax.plot([e["epoch"] for e in _h], [e["train_acc"] for e in _h], label="Entrenamiento (sin aumento)")
        _ax.plot([e["epoch"] for e in _h], [e["test_acc"] for e in _h], label="Prueba")
        _ax.set_xlabel("Época")
        _ax.set_ylabel("Accuracy")
        _ax.grid(alpha=0.3)
        _ax.legend()
        _out_dense = _fig
    else:
        _out_dense = "Falta results/vit_longer_dense.json"
    _out_dense
    return


@app.cell
def _(mo, runs, vit_dense):
    _vl = runs.get("ViT largo (100 ép.) / CIFAR-10")
    _rows = ["| % de pesos | Ticket desde $\\theta_0$ | Ticket con rewinding (época 5) | Aleatorio |", "|---:|---:|---:|---:|"]
    if _vl:
        for _i, _r in enumerate(_vl["ticket"]):
            _late = f"{_vl['ticket_late'][_i]['acc']:.2%}" if _i < len(_vl.get("ticket_late", [])) else "—"
            _rnd = f"{_vl['random'][_i]['acc']:.2%}" if _i < len(_vl["random"]) else "—"
            _rows.append(f"| {_r['weights_left']:.1%} | {_r['acc']:.2%} | {_late} | {_rnd} |")
    _last = vit_dense["history"][-1] if vit_dense else None
    mo.md((f"**Final de las {_last['epoch']} épocas:** prueba = {_last['test_acc']:.2%}, entrenamiento = {_last['train_acc']:.2%}.\n\n" if _last else "") + "**ViT largo (05), accuracy de prueba por porcentaje de pesos:**\n\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CNN: qué sobrevive por capa, poda por capa vs. global
    Mismas condiciones salvo el alcance de la poda (02: 20% de cada capa; 02b: un umbral de magnitud para toda la red). Se muestran las máscaras del ticket con 10.7% de los pesos totales (ronda 10). "Vacíos" = filtros convolucionales sin ningún peso vivo; en `fc3`, neuronas ocultas sin ningún peso hacia la salida.
    """)
    return


@app.cell
def _(mo, plt, torch):
    _dirs = {"Por capa (02)": "../checkpoints/02_cnn", "Global (02b)": "../checkpoints/02b_cnn_global"}
    _layers = ["conv.0", "conv.2", "conv.5", "conv.7", "linear.1", "linear.3", "linear.5"]
    _nice = ["conv1 (3→64)", "conv2 (64→64)", "conv3 (64→128)", "conv4 (128→128)", "fc1 (8192→256)", "fc2 (256→256)", "fc3 (256→10)"]
    _info = {}
    for _lab, _d in _dirs.items():
        _ck = torch.load(f"{_d}/ticket_r10_10.7pct.pt", map_location="cpu")
        _info[_lab] = {}
        for _n, _m in _ck["masks"].items():
            _key = _n.removesuffix(".weight")
            _row = {"keep": _m.float().mean().item(), "n": int(_m.sum())}
            if _m.dim() == 4:
                _row["dead"] = int((_m.flatten(1).sum(1) == 0).sum())
                _row["of"] = _m.shape[0]
            elif _key == "linear.5":
                _row["dead"] = int((_m.sum(0) == 0).sum())  # neuronas ocultas sin ningún peso a la salida
                _row["of"] = _m.shape[1]
            _info[_lab][_key] = _row
    _lines = ["| Capa | Por capa (02): % vivo | Pesos | Filtros / neuronas vacíos | Global (02b): % vivo | Pesos | Filtros / neuronas vacíos |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _k, _nm in zip(_layers, _nice):
        _cells = []
        for _lab in _dirs:
            _r = _info[_lab][_k]
            _cells += [f"{_r['keep']:.1%}", f"{_r['n']:,}", f"{_r['dead']}/{_r['of']}" if "dead" in _r else "—"]
        _lines.append(f"| {_nm} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _w = 0.38
    for _i, _lab in enumerate(_dirs):
        _ax.bar([x + _i * _w for x in range(7)], [_info[_lab][k]["keep"] * 100 for k in _layers], _w, label=_lab)
    _ax.axhline(10.7, color="gray", ls=":", lw=1)
    _ax.set_xticks([x + _w / 2 for x in range(7)])
    _ax.set_xticklabels(_nice, rotation=30, ha="right", fontsize=8)
    _ax.set_ylabel("Pesos vivos de la capa (%)")
    _ax.set_title("Ticket de la CNN con 10.7% de los pesos en total", fontsize=9)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("**Qué sobrevive en cada capa de la CNN** (ronda 10, 10.7% de los pesos en total). 'Vacíos' = filtros convolucionales sin ningún peso vivo; en `fc3`, neuronas ocultas sin ningún peso hacia la salida.\n\n" + "\n".join(_lines)),
               _fig,
               mo.md("*Cómo leerlo:* la línea punteada es el 10.7% global. Con poda por capa todas las barras azules están en ~10.7%, también las capas diminutas (conv1 con 1,728 pesos queda en 188). Con poda global las barras naranjas varían: las capas pequeñas y de pesos grandes casi no se podan y la capa enorme `fc1`, donde hay más pesos redundantes, absorbe la mayor parte de la poda.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CNN: tamaño de los pesos por capa vs. lo que sobrevive
    La poda global quita los pesos de menor magnitud de **toda** la red, así que lo que conserva cada capa depende de qué tan grandes son sus pesos. Se muestra la mediana de $|w|$ en la CNN densa entrenada (02b) y el porcentaje vivo de esa capa en el ticket global con 10.7% de los pesos. Las capas pequeñas (conv1, fc3) tienen pesos mucho más grandes que la capa enorme fc1; la poda por capa ignora eso y les exige perder lo mismo.
    """)
    return


@app.cell
def _(mo, plt, torch):
    import glob as _glob
    _names = ["conv.0", "conv.2", "conv.5", "conv.7", "linear.1", "linear.3", "linear.5"]
    _nice = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2", "fc3"]
    _b = torch.load("../checkpoints/02b_cnn_global/base.pt", map_location="cpu")
    _ck = torch.load(_glob.glob("../checkpoints/02b_cnn_global/ticket_r10_*.pt")[0], map_location="cpu")
    _med = [float(_b["dense"][k + ".weight"].abs().median()) for k in _names]
    _size = [_b["dense"][k + ".weight"].numel() for k in _names]
    _keep = [_ck["masks"][k + ".weight"].float().mean().item() * 100 for k in _names]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.bar(_nice, _med)
    _a1.set_yscale("log")
    _a1.set_ylabel("Mediana de |w| (red densa)")
    for _i, _n in enumerate(_size):
        _a1.text(_i, _med[_i] * 1.1, f"{_n:,}", ha="center", fontsize=7, rotation=90)
    _a1.set_ylim(top=0.5)
    for _m, _k, _l in zip(_med, _keep, _nice):
        _a2.scatter(_m, _k, color="C1")
        _a2.annotate(_l, (_m, _k), xytext=(4, 3), textcoords="offset points", fontsize=8)
    _a2.set_xscale("log")
    _a2.set_yscale("log")
    _a2.set_xlabel("Mediana de |w| (red densa)")
    _a2.set_ylabel("% vivo de la capa (ticket global, 10.7%)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"conv1: mediana {_med[0]:.3f} con {_size[0]:,} pesos (queda {_keep[0]:.0f}% vivo); fc1: mediana {_med[4]:.4f} con {_size[4]:,} pesos (queda {_keep[4]:.1f}% vivo). Cociente de medianas conv1/fc1: {_med[0] / _med[4]:.0f}x.")])
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cabezas de atención (solo ViT)
    El peso `qkv` de cada bloque tiene 3 partes (q, k, v) con una fila de cabeza por cada cabeza de atención. Se muestra la fracción viva de cada parte y cabeza al porcentaje objetivo. Una cabeza con ~0% está efectivamente apagada.
    """)
    return


@app.cell
def _(at_target, ckpt, np, plt):
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
        if _m.dim() < 2:
            _rows.append(f"| `{_n}` | {tuple(_m.shape)} | {int(_m.sum())} / {_m.numel()} ({_alive:.1%}) | - | - |")
            continue
        _out = _m.reshape(_m.shape[0], -1).any(dim=1)
        _inp = _m.transpose(0, 1).reshape(_m.shape[1], -1).any(dim=1)
        _rows.append(f"| `{_n}` | {tuple(_m.shape)} | {int(_m.sum())} / {_m.numel()} ({_alive:.1%}) | {int((~_out).sum())} / {len(_out)} | {int((~_inp).sum())} / {len(_inp)} |")
    mo.md(f"**Ticket con {_pct:.1f}% de los pesos (ronda {_r})**\n\n" + "\n".join(_rows))
    return (masks_t,)


@app.cell
def _(masks_t, np, plt):
    _names = [_n for _n, _v in masks_t.items() if _v.dim() >= 2]
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

    **Cómo leerlo:** arriba, la accuracy de cada clase por modelo; abajo, la diferencia contra el denso. Una barra fuera de la banda gris es una clase donde el modelo podado realmente difiere. La matriz de confusión (filas = clase real, columnas = predicha) muestra *qué* se confunde: fuera de la diagonal están los errores.
    """)
    return


@app.cell
def _(RUN_DIR, base, ckpt, curves, json, load_vit, os, predict):
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

    **Cómo leerlo:** más brillante = el `[CLS]` mira más ese parche (cada cuadro es de 4x4 píxeles). Si la atención cae sobre el objeto (cabeza, cuerpo, alas) el modelo parece usar la zona correcta; si se reparte uniforme, o se pega siempre al mismo borde para todas las imágenes, está usando algo menos específico de la imagen. La atención describe dónde pone peso el `[CLS]`, **no** explica por qué decide así: medida contra oclusión (tapar bloques y ver cuánto baja la probabilidad), su correlación de Spearman es ≈ 0 en denso, ticket, rewinding y aleatorio (`results/attention_vs_occlusion.json`).
    """)
    return


@app.cell
def _(
    at_target,
    base,
    ckpt,
    curves,
    device,
    fixed_x,
    load_vit,
    mean,
    plt,
    std,
    torch,
):
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

    **Cómo leerlo:** eje vertical logarítmico; una barra el doble de alta es un gradiente 2 veces mayor. Compara la misma capa entre curvas (ticket, rewinding, aleatorio), no capas distintas entre sí, porque la norma crece con el número de pesos. Un gradiente inicial menor sugiere que se arranca más cerca de un buen punto; no dice por sí solo qué modelo terminará mejor.
    """)
    return


@app.cell
def _(
    at_target,
    base,
    ckpt,
    curves,
    device,
    load_vit,
    nn,
    np,
    plt,
    sample_x,
    sample_y,
    torch,
):
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
