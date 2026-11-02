import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lottery Ticket Hypothesis a través de Arquitecturas
    **Proyecto Final - Deep Learning**

    > **Estado.** Todas las corridas terminaron (incluida `02b`, poda global de la CNN). Las tablas y gráficas se calculan de `results/*.json` y de `checkpoints/`. Los bloques marcados **Borrador** son interpretaciones propuestas, escritas para editarse con palabras propias; se apoyan en los números del reporte, pero las explicaciones causales son hipótesis, no resultados probados. Los textos descriptivos de las figuras dicen qué se ve y cómo leerlo.

    ## 1. Pregunta
    El Lottery Ticket Hypothesis (LTH) [1] dice que dentro de una red grande existe una subred pequeña que, entrenada desde los mismos pesos iniciales, alcanza una accuracy parecida a la de la red completa. En redes más grandes, regresar a la inicialización ya no basta y hay que rebobinar a un punto posterior del entrenamiento [2].

    Este proyecto prueba la idea en cuatro arquitecturas (MLP, CNN, ResNet y un ViT pequeño) y pregunta:

    1. **¿Existen winning tickets en cada arquitectura?** Es decir, ¿el ticket supera a la misma estructura con pesos aleatorios nuevos?
    2. **¿Hace falta rebobinar, y a qué punto?** Hipótesis: mientras más compleja la red, más tarde aparece el ticket.
    3. **¿Qué tan pequeña puede ser la red antes de dejar de funcionar?**

    ## 2. Método
    Para cada arquitectura se entrena la red densa, y se repite **IMP** (poda iterativa por magnitud): podar los pesos de menor magnitud, regresar los sobrevivientes a un punto de referencia, re-entrenar. Se comparan tres curvas de accuracy contra el porcentaje de pesos restantes:

    - **Ticket:** los sobrevivientes regresan a su valor en la inicialización (época 0).
    - **Ticket con rewinding:** regresan a su valor tras la época $k$ del entrenamiento denso (solo ResNet y ViT).
    - **Reinicialización aleatoria (control):** la misma máscara del ticket pero con pesos nuevos.

    | | MLP | CNN (02) | CNN global (02b) | ResNet | ViT (04) | ViT largo (05) |
    |---|---|---|---|---|---|---|
    | Red | LeNet-300-100 | Conv-4 | Conv-4 | CIFAR ResNet de 20 capas [1] | 6 bloques, dim 192 | igual que 04 |
    | Datos | MNIST | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 |
    | Optimizador | Adam 1.2e-3 | Adam 3e-4 | Adam 3e-4 | SGD 0.1, drops | AdamW 1e-3, wd 0.05 | igual |
    | Épocas por ronda | 5 | 30 | 30 | 77 | 20 | 100 |
    | Aumento de datos | no | no | no | flip + crop | flip + crop | + RandAugment, erasing, mixup/cutmix, label smoothing |
    | Poda | 20% por capa | 20% por capa | **20% global** | 20% global | 35% global | 35% global |
    | Rondas | 15 | 10 | 10 | 10 | 8 | 8 |
    | Rewind a época | - | - | - | 1 | 2 | 5 (fin del warmup) |

    **Cómo se hace una ronda de IMP** (diagrama en 2.1): (1) entrenar la red con la máscara actual; (2) quitar el 20% (35% en el ViT) de los pesos vivos con menor magnitud, **por capa** o **globalmente** (un solo umbral para toda la red); (3) regresar los pesos que sobreviven a su valor de referencia (época 0, o la época de rewind); (4) volver a entrenar. El control aleatorio usa la misma máscara con pesos nuevos.

    Las fuentes de cada hiperparámetro y las desviaciones respecto a ellas están en `REFERENCES.md`. Una sola semilla por experimento.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    import lth

    EXPERIMENTS = {
        "mlp_mnist": "MLP / MNIST",
        "cnn_cifar10": "CNN / CIFAR-10",
        "cnn_global_cifar10": "CNN poda global / CIFAR-10",
        "resnet18_cifar10": "ResNet / CIFAR-10",
        "vit_cifar10": "ViT (20 ép.) / CIFAR-10",
        "vit_longer": "ViT largo (100 ép.) / CIFAR-10",
    }

    runs, partial = {}, []
    for _file, _name in EXPERIMENTS.items():
        _path = Path("../results") / f"{_file}.json"
        if _path.exists():
            runs[_name] = json.loads(_path.read_text())

    # El ViT largo escribe su JSON hasta el final; mientras tanto se leen las curvas de los checkpoints
    _vl = EXPERIMENTS["vit_longer"]
    _ck = Path("../checkpoints/05_vit_longer")
    if _vl not in runs and _ck.exists():
        _alias = {"ticket": "ticket", "late": "ticket_late", "random": "random"}
        _cur = {}
        for _tag, _r, _pct, _acc, _ in lth.list_checkpoints(str(_ck)):
            _cur.setdefault(_alias[_tag], []).append({"round": _r, "weights_left": _pct / 100, "acc": _acc})
        if "ticket" in _cur:
            runs[_vl] = {k: sorted(v, key=lambda r: r["round"]) for k, v in _cur.items()}
            partial.append(_vl)

    _dense_path = Path("../results/vit_longer_dense.json")
    vit_dense = json.loads(_dense_path.read_text()) if _dense_path.exists() else None
    return json, mo, partial, plt, runs, vit_dense


@app.cell
def _(mo, partial, runs):
    mo.md(
        "**Estado de los datos:** " + ", ".join(f"`{n}`" for n in runs)
        + (f".  **Parcial (corrida en curso):** {', '.join(partial)}." if partial else ".")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Arquitecturas e IMP
    Diagramas de las cuatro redes (formas de los tensores entre paréntesis) y del ciclo de IMP. Las cifras de parámetros son de las capas que se podan (pesos, sin sesgos). Dibujados con matplotlib para este reporte.
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
    _row(_ax, 5.4, [("Imagen\n(3,32,32)", 1.1), ("Conv 3→16\n+BN+ReLU", 1.1), ("3 bloques\n16 canales\n(32×32)", 1.25), ("3 bloques\n32 canales\n(16×16)", 1.2), ("3 bloques\n64 canales\n(8×8)", 1.2), ("AvgPool\n+ Linear 64→10\n(sin poda)", 1.1), ("10 clases", 0.9)],
         "ResNet de 20 capas [1] · CIFAR-10 · ~270k pesos podables (solo convoluciones)", {1: _prune, 2: _prune, 3: _prune, 4: _prune})
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
    ## 3. Resultados
    ### 3.1 Curvas por arquitectura
    Accuracy de prueba contra el porcentaje de pesos restantes (eje x logarítmico e invertido: la red se hace más pequeña hacia la derecha).

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
    ### 3.2 Accuracy relativa a la red densa
    Se resta la accuracy densa de cada curva para comparar arquitecturas con bases distintas. 0 = la subred iguala a la red completa.
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
    ### 3.3 Resumen
    - **Mín. pesos (ticket / aleatorio):** la red más pequeña que queda a menos de 1 punto de la accuracy densa.
    - **Ventaja del ticket:** diferencia promedio entre el ticket y la reinicialización aleatoria sobre las rondas podadas.
    - **Ganancia del rewinding:** diferencia promedio entre el ticket con rewinding y el clásico.

    Con una sola semilla y ~10,000 imágenes de prueba, diferencias de ~0.5 puntos o menos no son concluyentes.
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
    ## 4. Resultados por arquitectura
    **MLP / MNIST.** El ticket se mantiene entre 97.2% y 98.2% hasta 3.5% de los pesos, a la par de la red densa (97.7%), mientras que la reinicialización aleatoria cae a 95.7%. Regresar a la inicialización ya basta; no se probó el rewinding aquí. La máscara de la primera capa, dibujada como imagen de 28x28, poda sobre todo el borde (128 de los 784 pixeles de entrada quedan sin ningún peso), y en la segunda capa 54 de las 100 neuronas quedan sin pesos; es esperable porque los pixeles que siempre valen 0 nunca reciben gradiente.

    **CNN / CIFAR-10 (poda por capa, 02).** En la primera corrida el ticket supera a la red densa (73.6%) con un máximo de 75.9% a 64% de los pesos, y le gana a la reinicialización aleatoria entre 80% y ~33%. A 10.7% el ticket colapsa a 49.7%, por debajo del control aleatorio (61.0%). La repetición de la corrida reproduce el colapso (53.0% vs. 61.2%), así que no es ruido. Al inspeccionar la máscara, en la primera convolución 7 de 64 filtros quedan completamente vacíos y 102 de las 256 neuronas ocultas no tienen ningún peso hacia la salida.

    **CNN / CIFAR-10 (poda global, 02b).** Se repitió el experimento cambiando solo la poda: un umbral de magnitud para toda la red en lugar de 20% por capa. El colapso desaparece: a 10.7% de los pesos el ticket tiene **75.7%** (red densa: 74.0%) y el control aleatorio 69.4%, es decir el ticket queda 6.2 puntos *por encima* del aleatorio, cuando con poda por capa quedaba 8 puntos por debajo. Las máscaras explican la diferencia (tabla y figura abajo): con poda global la primera convolución conserva 88% de sus pesos (1,523 contra 188), no hay filtros vacíos y no hay neuronas ocultas sin salida; el costo lo paga la primera capa lineal, que pasa a 4.8%. Esto apoya que la causa del colapso era la poda por capa, aunque sigue siendo una sola corrida por configuración y los dos experimentos difieren también en la semilla de la inicialización.
    """)
    return


@app.cell
def _(mo, plt):
    import torch
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
    **ResNet / CIFAR-10.** Con lr 0.1 el ticket que regresa a la época 0 **no** supera al aleatorio (86.3% vs. 86.5% con 10.7% de los pesos), igual que en [1]. Rebobinar a la época 1 recupera casi toda la accuracy: 89.6% con 10.7% de los pesos, contra 90.9% de la red densa.

    **ViT / CIFAR-10 (20 épocas).** El ticket se mantiene en 82-83% de 42% a 7.5% de los pesos, por encima de la red densa (78.9%), mientras que el control aleatorio cae a 65.9% con 3.2%. Rebobinar a la época 2 queda igual o ligeramente por debajo del ticket clásico. El baseline de 78.9% probablemente está subentrenado (20 épocas), así que el resultado "el ticket supera a la red densa" no se puede atribuir a una regularización sin revisar eso primero. [5] reporta que los tickets a nivel de peso son difíciles de encontrar en ViTs, así que este resultado se presenta con reserva.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ViT con entrenamiento más largo (05)
    El baseline de 04 (78.9%) parecía subentrenado. En 05 el ViT denso se entrena 100 épocas con aumento de datos más agresivo y label smoothing, y se mide la accuracy de entrenamiento (sin aumento) contra la de prueba en cada época para ver si hay sobreajuste.
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
        _out2 = _fig
    else:
        _out2 = "Falta results/vit_longer_dense.json (el baseline denso todavía no termina)."
    _out2
    return


@app.cell
def _(mo, vit_dense):
    if vit_dense:
        _h = vit_dense["history"][-1]
        _out3 = mo.md(
            f"Al final de las {_h['epoch']} épocas: prueba = **{_h['test_acc']:.2%}**, entrenamiento = {_h['train_acc']:.2%} "
            f"(brecha de {(_h['train_acc'] - _h['test_acc']) * 100:.1f} puntos), contra 78.9% de prueba en 04. "
            "Una brecha de unos pocos puntos con accuracy de prueba todavía subiendo indica un modelo que no está fuertemente sobreajustado."
        )
    else:
        _out3 = mo.md("**[PENDIENTE]**")
    _out3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Resultados.** Con 100 épocas la red densa llega a 86.0% (antes 78.9%). El ticket sigue por encima de la red densa hasta ~12% de los pesos, así que el resultado de 04 no era solo un baseline débil. Se midió también el denso desde otro arranque (ronda 0 del control aleatorio): 85.7%, es decir ~0.2 puntos de diferencia solo por la inicialización.

    | % de pesos | Ticket desde θ₀ | Ticket con rewinding (época 5) | Aleatorio |
    |---:|---:|---:|---:|
    | 100 (denso) | 86.0 | 86.0 | 85.7 |
    | 27.5 | 89.0 | 89.5 | 84.1 |
    | 11.6 | 88.1 | 88.4 | 78.0 |
    | 7.5 | 83.1 | 86.6 | 75.3 |
    | 4.9 | 80.4 | 83.7 | 71.5 |
    | 3.2 | 77.3 | 78.4 | 68.5 |

    - El ticket le gana al control aleatorio en todos los puntos, y la diferencia crece al podar más (de 5 puntos a 27.5% a ~9 puntos a 3.2%).
    - El rewinding no cambia nada arriba de ~10% de los pesos; por debajo, mejora al ticket clásico (+3.4 puntos a 7.5%, +3.3 a 4.9%, +1.2 a 3.2%), en una sola corrida.
    - Por encima de la red densa: ticket clásico hasta 11.6%; con rewinding hasta 7.5%. [5] reporta que los tickets de peso son difíciles en ViTs; aquí el efecto puede deberse en parte a regularización por la poda, y con una semilla no se puede separar.

    **Borrador de interpretación** (editar con tus palabras): ¿por qué el rewinding solo ayuda en los puntos muy esparcidos?

    - Con 12% o más de los pesos, la subred todavía es lo bastante grande para recuperarse desde cualquier inicialización razonable; ahí ticket, rewinding y la red densa quedan dentro de ~1 punto, que es el ruido de una sola corrida.
    - Con 7.5% y 4.9% de los pesos la red tiene poca capacidad y su entrenamiento es más sensible. Frankle et al. [2] ven que, en redes grandes, el ticket solo es estable si se arranca después de unas pocas iteraciones, cuando el entrenamiento ya salió de la zona más inestable. Las 5 épocas del rewinding coinciden con el final del warmup; es compatible con esa idea, pero no se probó (haría falta variar el punto de rewind).
    - 04 rewindeaba en la época 2 de 20 y tenía un baseline subentrenado (78.9%); el ticket ya superaba a la red densa por mucho, así que no había margen para ver una ventaja. Además 04 no tiene los puntos 7.5% y 4.9% donde 05 muestra la ganancia. Las dos corridas no son comparables punto por punto.
    - Cautela: +3.4 puntos con una semilla está por encima del ruido de ±0.6, pero no se repitió.

    ## 6. Inspección de los tickets
    Checkpoints del ViT largo, comparados al mismo porcentaje de pesos (11.6%, ronda 5) con `06_inspection.py`. El código de las figuras fue generado con IA; los pesos y las máscaras son los de las corridas. Las descripciones dicen lo que se ve y cómo leerlo.

    **Guía rápida de las vistas** (una por figura):

    | Vista | Qué es | Cómo leerla | Cuidado |
    |---|---|---|---|
    | Pesos vivos por capa (mapa de calor) | Para cada capa y cada ronda, la fracción de sus pesos que sobrevive a la poda. Clara = casi todos vivos, oscura = casi vacía. | Una capa que se oscurece rápido es una capa que la poda considera prescindible (sus pesos son pequeños). Con poda global las filas se separan; con poda por capa serían parejas. | Una capa con muy pocos pesos puede quedar desconectada del resto. |
    | Accuracy por clase | Para cada una de las 10 clases, el porcentaje de acierto del modelo. La gráfica de abajo resta la del modelo denso. | Barras positivas = el modelo podado acierta más que el denso en esa clase. La banda gris (±2.5 puntos) es el ruido estadístico de 1,000 imágenes por clase: lo que cae dentro no se interpreta. | Un solo modelo por curva; no hay barras de error entre corridas. |
    | Mapas de atención | Cuánto "mira" el token [CLS] (el que decide la clase) a cada uno de los 64 parches de 4×4 píxeles, en el último bloque, promediando las 3 cabezas. Se muestra sobre la imagen: más brillante = más peso. | Si la atención se concentra en el objeto (cabeza del gato, motor del avión), el modelo parece usar la zona correcta; si se reparte por el fondo o se pega a una esquina, usa otra cosa. Se compara denso, ticket, rewinding y aleatorio sobre las mismas 5 imágenes. | Atención no es explicación: aquí se midió y su correlación con la oclusión es ≈ 0 (denso 0.05, ticket 0.06, aleatorio 0.12), así que es descripción, no prueba. |
    | Entropía de la atención | Qué tan repartida está la atención (alta = uniforme, baja = concentrada en pocos parches). | Compara entre modelos y entre rondas: una entropía que cae mucho al podar indica atención más enfocada (o más degenerada). | No dice si la atención es *correcta*. |
    | Normas de gradiente | Qué tan grande es el gradiente de la pérdida sobre los pesos vivos, por tipo de capa, en el paso 0 (antes de entrenar) y al final. Escala logarítmica. | Un gradiente más grande en el paso 0 significa que la inicialización está "más lejos" de un buen punto y recibe actualizaciones grandes; si el ticket y el aleatorio se parecen, la inicialización no se distingue por su gradiente inicial. | La norma depende del tamaño de la capa; compara la misma capa entre curvas, no capas distintas. |
    | IoU ticket vs. rewinding (06) | Qué fracción de los pesos vivos comparten las dos máscaras (intersección / unión). 1 = idénticas. | Un valor alto significa que ambos métodos encuentran casi la misma subred; uno bajo, que hay muchas subredes distintas que funcionan. | Para máscaras aleatorias de la misma densidad el IoU esperado ya es bajo; hay que compararlo con ese nivel. |
    | Matriz de confusión (06) | Filas: clase real; columnas: clase predicha. | La diagonal son los aciertos; los números fuera de ella dicen qué clases se confunden (gato↔perro es el caso típico). | Se lee mejor al comparar la del denso con la del ticket. |
    """
    )
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("**Pesos que sobreviven por tipo de capa** (ticket, todas las rondas, promedio ponderado sobre los 6 bloques)."),
        mo.image("../presentation/img/heat.png"),
        mo.md("Con poda global las capas no pierden parejo. En la última ronda (3.2% de los pesos en total) `head` conserva 21%, `patch_embed` 7%, `qkv` 4.5%, `proj` 2.6%, `mlp.0` 2.7% y `mlp.2` 1.3%. Además, las capas LayerNorm también entran a la poda."),
        mo.md("**Accuracy por clase** (diferencia contra el denso, 11.6% de los pesos; la banda gris es el ruido de ±2.5 puntos por clase)."),
        mo.image("../presentation/img/cls.png"),
        mo.md("El ticket queda entre -0.4 y +4.5 puntos del denso (mejor: perro) y el rewinding entre +0.7 y +5.5 (mejor: gato). El control aleatorio pierde en todas las clases, hasta 14.6 puntos en pájaro. Las clases con menor accuracy en el denso son gato (70.0%) y perro (77.7%)."),
        mo.md("**Mapas de atención** del `[CLS]` en el último bloque (promedio de cabezas), 5 imágenes de prueba."),
        mo.image("../presentation/img/attn.png"),
        mo.md("En las filas denso, ticket y rewinding la atención se reparte por distintas zonas de la imagen. En el control aleatorio (misma máscara que el ticket, otra inicialización) se concentra en la parte inferior de las 5 imágenes."),
        mo.md("**Gradientes en el paso 0** (norma sobre los pesos vivos, por tipo de capa)."),
        mo.image("../presentation/img/grad.png"),
        mo.md("El rewinding arranca con normas menores en las capas internas que el ticket (por ejemplo `qkv`: 0.10 contra 0.29) y el aleatorio queda cerca del ticket; en `head` las normas son 1.16 (ticket), 0.61 (rewinding) y 1.21 (aleatorio)."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Borrador de interpretación** (editar con tus palabras; las causas son hipótesis):

    - **`head` y `patch_embed` conservan más pesos.** Con poda global se quita el 35% de menor magnitud *de toda la red*, así que sobreviven más las capas cuyos pesos son grandes. `head` y `patch_embed` son capas pequeñas y con pocas entradas por neurona, y están en los extremos de la red, donde cada peso individual carga más información (no hay redundancia, como en las 6 capas MLP del medio). Es una explicación plausible, no probada; confirmarla pediría mirar la distribución de magnitudes por capa. Es el mismo patrón que en la CNN global, donde la primera convolución conserva 88%.
    - **Atención del control aleatorio concentrada abajo.** El control comparte la máscara con el ticket y solo cambia los valores iniciales; si la atención cambia, la inicialización (o el camino de entrenamiento que induce) determina hacia dónde mira la red, no solo la estructura. No sabemos cuál de los dos factores pesa más; con 5 imágenes y una corrida es una observación, no un resultado.
    - **Rewinding con gradientes menores al inicio.** A partir de la época 5 los pesos ya salieron del arranque más abrupto y están en una región con menos pendiente; es coherente con que 5 épocas son el fin del warmup. No se midió si esos gradientes menores *causan* la mejora a baja densidad.
    - **Ventaja por clase.** El ticket queda dentro del ruido (±2.5) en la mayoría de las clases; la ventaja no viene de una sola clase, pero con 1,000 imágenes por clase tampoco se puede afirmar que sea pareja. Las clases difíciles del denso (gato, perro) son donde más se mueven los modelos podados.

    La mismas vistas, con más detalle (cabezas de atención, IoU entre el ticket y el rewinding, forma de las máscaras, entropía de la atención), están en `06_inspection.html`.

    ## 7. Discusión
    **Qué sí apoyan los resultados.**
    - En el MLP y la CNN, regresar a la inicialización basta: el ticket iguala o supera a la red densa y le gana al control aleatorio. En la CNN esto es cierto hasta 10.7% de los pesos solo con poda global; con poda por capa el ticket colapsa en ese último punto. **La forma de podar cambió el resultado** más que la arquitectura.
    - En la ResNet con lr 0.1, regresar a la época 0 no basta y rebobinar a la época 1 sí, igual que en [1] y [2].
    - En el ViT, el ticket clásico ya funciona sin rewinding; el control aleatorio con la misma estructura queda claramente por debajo.

    **Sobre "más compleja la red, más tarde aparece el ticket".** Los datos son compatibles de forma cualitativa: el rewinding hace falta en la ResNet y no en el MLP. Pero no es una prueba. Cada arquitectura tiene como mucho uno o dos puntos de rewind, así que no hay una curva de accuracy contra la época $k$. Además las arquitecturas difieren en más que su complejidad: datos distintos (MNIST vs. CIFAR-10), optimizadores y schedules de lr distintos (el ViT tiene warmup, la ResNet empieza con lr 0.1 sin él), poda por capa vs. global, tasas de poda y número de épocas. Una posible explicación de que el ViT no necesite rewinding es su warmup, que estabiliza el inicio del entrenamiento; esa hipótesis no se probó.

    **Limitaciones.**
    - **Una sola semilla por experimento** y ~10,000 imágenes de prueba: diferencias de ~1 punto o menos son ruido, y no hay barras de error ni pruebas de significancia.
    - **No se barrió el punto de rewind** $k$ (y por ello no se puede decir *cuándo* se fija el ticket en cada arquitectura).
    - **Presupuesto de cómputo:** todo el proyecto usó unas 10 horas de GPU (RTX 4070 SUPER, ~4 kWh); el ViT largo tarda ~5 horas por corrida, un barrido de $k$ sobre cuatro arquitecturas con varias semillas queda fuera del alcance.
    - **Desviaciones de las fuentes** (épocas por ronda, tasas de poda, learning rate del ViT sin ajustar): ver `REFERENCES.md`.
    - Tampoco se probó si un ticket encontrado en un dataset sirve en otro (parte opcional de la propuesta).

    ## 8. Preguntas para la discusión
    Preguntas abiertas que los datos permiten plantear pero no cerrar. Cada una indica qué experimento la resolvería.

    **Sobre la hipótesis "más compleja, más tarde".**
    1. La ResNet necesita rewinding a la época 1 con lr 0.1 y sin warmup; el ViT con warmup no lo necesitó a 11.6% de los pesos y solo lo aprovechó por debajo de ~8%. ¿El factor es la arquitectura o la estabilidad del arranque del optimizador? *(Repetir la ResNet con warmup y el ViT sin él.)*
    2. En el ViT largo el rewinding no ayudó con poco podado pero sí con mucho. ¿El punto de rewind necesario depende del nivel de poda, no solo de la arquitectura? *(Barrido de $k$ a dos o tres sparsidades fijas.)*
    3. ¿Es 1 época "tarde" en una ResNet y 5 épocas "tarde" en un ViT, o conviene medirlo en fracción del entrenamiento (1/30 contra 5/100)? ¿Con qué unidad se compara el $k$ entre arquitecturas?

    **Sobre qué es un ticket.**
    4. Que el control aleatorio (misma máscara, pesos nuevos) falle por mucho indica que no es solo la estructura. ¿Qué parte la aporta la inicialización exacta y qué parte la *escala* de los pesos que sobrevivieron? *(Reinicializar con los mismos valores permutados por capa.)*
    5. En el ViT largo el ticket supera a la red densa con ~12% de los pesos. ¿Es un efecto de regularización por poda o un baseline todavía mejorable? *(Más épocas o más aumento al denso; varias semillas.)*
    6. En 04 el ticket superaba al denso (subentrenado) y en 05 también (mejor entrenado), pero [5] dice que estos tickets son difíciles en ViTs. ¿Qué condiciones del recipe (aumento de datos, tamaño del modelo, poda global) explican la diferencia?

    **Sobre la CNN y la poda por capa.**
    7. El colapso a 10.7% se repitió y coincide con filtros y neuronas completamente vacíos. ¿Sería evitable con poda global, que puede podar menos las capas pequeñas? *(Respondida por 02b: sí, a 10.7% el ticket pasa de 53.0% a 75.7%, sin filtros ni neuronas vacíos. Queda abierto si es la poda global en sí o simplemente que la primera conv conserve más pesos; se separaría fijando un mínimo por capa en la poda por capa.)*
    8. ¿Una capa con 7 de 64 filtros vacíos es un problema de capacidad o solo desperdicio de pesos? *(Podar manteniendo un mínimo por filtro.)*

    **Sobre las máscaras.**
    9. ¿Por qué la atención del control aleatorio se concentra abajo si comparte la máscara con el ticket? ¿Se puede usar la atención para explicarlo? (Medido: no, correlación ≈ 0 con oclusión.)
    10. ¿Qué tanto se parecen las máscaras del ticket y del rewinding (IoU, ver 06)? Si se parecen poco pero ambos funcionan, ¿hay muchos tickets distintos?

    **Sobre la metodología.**
    11. Con una sola semilla y ~10,000 imágenes de prueba, ¿qué diferencias son creíbles? (Un intervalo binomial del 95% sobre una accuracy de ~88% mide ±0.6 puntos; las diferencias entre ticket y aleatorio de 5-9 puntos lo superan, pero la de ticket contra denso a 7.5% no.)
    12. ¿Cómo afecta que la poda esté global en unos modelos y por capa en otros a la comparación entre arquitecturas?

    ### 8.1 Borradores de respuesta
    Respuestas tentativas, con lo que los datos permiten decir. Están pensadas como punto de partida para reescribir.

    1. **Arquitectura o estabilidad del arranque.** Los datos no separan las dos. Lo que se observa: ResNet (lr 0.1, sin warmup) necesita rewinding; el ViT (con warmup) no lo necesita a ≥12% y sí lo aprovecha por debajo de ~8%. Es compatible con que el warmup haga el trabajo del rewinding, pero también con que el ViT sea más tolerante. Confundidos: optimizador (SGD vs. AdamW), datos, tasa de poda. Se resolvería repitiendo con warmup/sin warmup.
    2. **El punto de rewind depende del nivel de poda.** Los datos del ViT largo lo sugieren (sin efecto a ≥12%, +3.4 puntos a 7.5%), pero solo hay un punto de rewind (época 5) y una semilla. Es una pista, no una conclusión.
    3. **Unidad del punto de rewind.** Medido en fracción del entrenamiento, 1 época de 77 (1.3%) en la ResNet y 5 de 100 (5%) en el ViT son del mismo orden. Medido en iteraciones: 391 contra ~1,950. No hay una unidad natural; lo razonable es reportar las tres (épocas, iteraciones, fracción) y barrer $k$.
    4. **Estructura vs. valores.** El control aleatorio (misma máscara) pierde 5-9 puntos en el ViT largo y ~2 en el MLP, así que la estructura sola no basta. Falta separar *qué* de los valores iniciales importa: los valores exactos o su escala. Reinicializar permutando los valores del ticket dentro de cada capa mantendría la escala y rompería la asignación.
    5. **Ticket > denso en el ViT.** El denso de 100 épocas llega a 86.0% y el ticket a 89.0% con 27.5% de los pesos, así que no es solo un baseline débil. Hipótesis restante: la poda actúa como regularizador (menos parámetros, menos sobreajuste); apoya que hay brecha entrenamiento-prueba. Se probaría con más aumento de datos al denso o con varias semillas.
    6. **Tickets en ViT según [5].** [5] reporta que los tickets a nivel de peso son difíciles de encontrar en ViTs; aquí el ViT es pequeño (1.8M pesos), con aumento de datos fuerte y poda global. No se pueden separar esos factores con los datos; la diferencia podría ser de tamaño de modelo o de receta, y hay que presentarlo con reserva.
    7. **Poda global y CNN.** Respondida: con poda global el ticket a 10.7% pasa de 53.0% a 75.7%, y las máscaras no tienen filtros ni neuronas vacíos. Es evidencia de que el colapso era por la poda por capa, no por la CNN; no separa "poda global" de "cada capa conserva más pesos" (se haría con un mínimo por capa).
    8. **Filtros vacíos.** 7 de 64 filtros vacíos en la primera convolución es capacidad perdida (esas salidas son constantes), y se suma a que 102 de las 256 neuronas ocultas no llegan a la salida. Con 10.7% de los pesos totales esta red perdía capacidad real, no solo pesos redundantes. Probar con un mínimo de pesos por filtro lo confirmaría.
    9. **Atención del control aleatorio.** Los mapas son distintos a los del ticket, pero la atención cruda tiene correlación ≈ 0 con la oclusión en todos los modelos, así que no explica por qué falla el control; solo se puede decir que la inicialización cambia los mapas.
    10. **Ticket vs. rewinding.** Comparar el IoU de 06 contra el IoU esperado entre dos máscaras al azar de la misma densidad. Si es claramente mayor, ambos métodos redescubren parte de la misma estructura; si es bajo y ambos funcionan, hay muchos tickets distintos.
    11. **Qué diferencias son creíbles.** El intervalo binomial del 95% con 10,000 imágenes y ~88% es ±0.6 puntos. Ticket vs. aleatorio (5-9 puntos), ticket vs. denso con 27.5% de los pesos (+3 puntos) y colapso de la CNN (−8 puntos) superan ese intervalo; ticket vs. denso con 7.5% de los pesos no. Aun así el ruido entre corridas (semilla) no se midió y puede ser mayor que el binomial.
    12. **Poda global vs. por capa entre arquitecturas.** El resultado de la CNN muestra que no es un detalle: cambia el resultado. Comparar el MLP (por capa) con el ViT (global) mezcla arquitectura con tipo de poda. Lo limpio sería repetir cada arquitectura con ambos tipos de poda.

    ## 9. Conclusiones y trabajo futuro
    1. Se encontraron winning tickets en el MLP, la CNN y el ViT (también con 100 épocas y mejor baseline); en la ResNet solo con rewinding. El colapso del ticket de la CNN a 10.7% se debió a la poda por capa: con poda global desaparece (75.7% contra 53.0%).
    2. Cuánto hay que rebobinar parece depender de la arquitectura y del recipe de entrenamiento, pero con los datos actuales no se puede separar un efecto del otro.
    3. Trabajo futuro: barrido del punto de rewind $k$ en las cuatro arquitecturas con varias semillas; repetir el ViT con y sin warmup; transferencia de tickets entre datasets.

    ## Referencias
    1. Frankle & Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR 2019. https://arxiv.org/abs/1803.03635
    2. Frankle, Dziugaite, Roy & Carbin, *Stabilizing the Lottery Ticket Hypothesis*, 2019. https://arxiv.org/abs/1903.01611
    3. Renda, Frankle & Carbin, *Comparing Rewinding and Fine-tuning in Neural Network Pruning*, ICLR 2020. https://arxiv.org/abs/2003.02389
    4. Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021. https://arxiv.org/abs/2012.12877
    5. Shen, Kong, Qin, Dong, Yuan, Meng, Tang, Ma & Wang, *Data Level Lottery Ticket Hypothesis for Vision Transformers*, IJCAI 2023. https://arxiv.org/abs/2211.01484
    """)
    return


if __name__ == "__main__":
    app.run()
