import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lottery Ticket Hypothesis a través de Arquitecturas
    **Proyecto Final - Deep Learning**

    > **Borrador.** Las secciones marcadas con **[PENDIENTE]** dependen de corridas que todavía no terminan (ViT largo, repetición de la CNN, inspección de máscaras). El resto se calcula directamente de `results/*.json`, así que al terminar las corridas las cifras se actualizan solas; el texto que las interpreta hay que revisarlo.

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

    | | MLP | CNN | ResNet | ViT (04) | ViT largo (05) |
    |---|---|---|---|---|---|
    | Red | LeNet-300-100 | Conv-4 | CIFAR ResNet de 20 capas [1] | 6 bloques, dim 192 | igual que 04 |
    | Datos | MNIST | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 |
    | Optimizador | Adam 1.2e-3 | Adam 3e-4 | SGD 0.1, drops | AdamW 1e-3, wd 0.05 | igual |
    | Épocas por ronda | 5 | 30 | 77 | 20 | 100 |
    | Aumento de datos | no | no | flip + crop | flip + crop | + RandAugment, erasing, mixup/cutmix, label smoothing |
    | Poda | 20% por capa | 20% por capa | 20% global | 35% global | 35% global |
    | Rondas | 15 | 10 | 10 | 8 | 8 |
    | Rewind a época | - | - | 1 | 2 | 5 (fin del warmup) |

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
    **MLP / MNIST.** El ticket se mantiene entre 97.5% y 98.2% hasta 3.5% de los pesos, a la par de la red densa (97.7%), mientras que la reinicialización aleatoria cae a 96.0%. Regresar a la inicialización ya basta; no se probó el rewinding aquí.

    **CNN / CIFAR-10.** En la primera corrida el ticket supera a la red densa (73.6%) con un máximo de 75.9% a 64% de los pesos, y le gana a la reinicialización aleatoria entre 80% y ~33%. A 10.7% el ticket colapsa a 49.7%, por debajo del control aleatorio (61.0%). La repetición de la corrida reproduce el colapso (53.0% vs. 61.2%), así que no es ruido. La CNN poda **por capa**, de modo que todas las capas conservan ~10.7% de sus pesos; al inspeccionar la máscara (06), en la primera convolución 7 de 64 filtros quedan completamente vacíos y 102 de las 256 neuronas ocultas no tienen ningún peso hacia la salida. Esa es una explicación posible, no probada; el notebook 02b repite el experimento con poda global.

    **ResNet / CIFAR-10.** Con lr 0.1 el ticket que regresa a la época 0 **no** supera al aleatorio (86.3% vs. 86.5% con 10.7% de los pesos), igual que en [1]. Rebobinar a la época 1 recupera casi toda la accuracy: 89.6% con 10.7% de los pesos, contra 90.9% de la red densa.

    **ViT / CIFAR-10 (20 épocas).** El ticket se mantiene en 82-83% de 42% a 7.5% de los pesos, por encima de la red densa (78.9%), mientras que el control aleatorio cae a 65.9% con 3.2%. Rebobinar a la época 2 queda igual o ligeramente por debajo del ticket clásico. El baseline de 78.9% probablemente está subentrenado (20 épocas), así que el resultado "el ticket supera a la red densa" no se puede atribuir a una regularización sin revisar eso primero. [5] reporta que los tickets a nivel de peso son difíciles de encontrar en ViTs, así que este resultado se presenta con reserva.
    """)
    return


@app.cell
def _(mo, runs):
    _c = runs.get("CNN / CIFAR-10")
    if _c:
        _t, _r = _c["ticket"][-1], _c["random"][-1]
        _out = mo.md(
            f"**Corrida actual de la CNN, punto más pequeño** ({_t['weights_left']:.1%} de los pesos): "
            f"ticket = {_t['acc']:.2%}, aleatorio = {_r['acc']:.2%}. "
            + ("El ticket sigue por debajo del aleatorio (el colapso se repite)." if _t["acc"] < _r["acc"] else "El ticket no cae por debajo del aleatorio (el colapso no se repite).")
        )
    else:
        _out = mo.md("Sin resultados de la CNN.")
    _out
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
    **[PENDIENTE]** Curvas de IMP del ViT largo (ticket, rewinding a la época 5, aleatorio), que aparecen en la sección 3 al terminar la corrida. Preguntas que responden:

    - ¿El ticket sigue por encima de la red densa ahora que el baseline mejoró? Si sí, no se explica solo por un baseline débil; si no, el efecto de 04 era del baseline.
    - ¿Rebobinar a la época 5 (fin del warmup) cambia algo respecto al ticket clásico, como en 04 con la época 2?
    - ¿El control aleatorio sigue cayendo mucho más que el ticket?

    ## 6. Inspección de los tickets
    **[PENDIENTE]** Se ejecuta `06_inspection.py` sobre los checkpoints del ViT largo (y de la CNN, la sección de máscaras) y se incluyen aquí las figuras:

    - **Pesos que sobreviven por capa y por cabeza de atención.** Qué capas (`qkv`, MLP, `patch_embed`, `head`) pierden más pesos con poda global, y si alguna cabeza queda apagada. Para la CNN, si el punto de 10.7% se explica por una capa pequeña casi vacía.
    - **Accuracy por clase:** denso vs ticket vs aleatorio, con ruido de ~±2.5 puntos por clase.
    - **Mapas de atención:** atención del `[CLS]` de denso, ticket, rewinding y aleatorio, y entropía de la atención contra sparsidad.
    - **Gradientes por capa:** al inicio del entrenamiento (inicialización con la máscara aplicada) y al final.

    ## 7. Discusión
    **Qué sí apoyan los resultados.**
    - En el MLP y (hasta cierto punto) la CNN, regresar a la inicialización basta: el ticket iguala o supera a la red densa y le gana al control aleatorio.
    - En la ResNet con lr 0.1, regresar a la época 0 no basta y rebobinar a la época 1 sí, igual que en [1] y [2].
    - En el ViT, el ticket clásico ya funciona sin rewinding; el control aleatorio con la misma estructura queda claramente por debajo.

    **Sobre "más compleja la red, más tarde aparece el ticket".** Los datos son compatibles de forma cualitativa: el rewinding hace falta en la ResNet y no en el MLP. Pero no es una prueba. Cada arquitectura tiene como mucho uno o dos puntos de rewind, así que no hay una curva de accuracy contra la época $k$. Además las arquitecturas difieren en más que su complejidad: datos distintos (MNIST vs. CIFAR-10), optimizadores y schedules de lr distintos (el ViT tiene warmup, la ResNet empieza con lr 0.1 sin él), poda por capa vs. global, tasas de poda y número de épocas. Una posible explicación de que el ViT no necesite rewinding es su warmup, que estabiliza el inicio del entrenamiento; esa hipótesis no se probó.

    **Limitaciones.**
    - **Una sola semilla por experimento** y ~10,000 imágenes de prueba: diferencias de ~1 punto o menos son ruido, y no hay barras de error ni pruebas de significancia.
    - **No se barrió el punto de rewind** $k$ (y por ello no se puede decir *cuándo* se fija el ticket en cada arquitectura).
    - **Presupuesto de cómputo:** el ViT largo ya tarda varias horas por corrida; un barrido de $k$ sobre cuatro arquitecturas con varias semillas queda fuera del alcance.
    - **Desviaciones de las fuentes** (épocas por ronda, tasas de poda, learning rate del ViT sin ajustar): ver `REFERENCES.md`.
    - Tampoco se probó si un ticket encontrado en un dataset sirve en otro (parte opcional de la propuesta).

    ## 8. Conclusiones y trabajo futuro
    1. Se encontraron winning tickets en el MLP, la CNN (con reserva en el punto más pequeño) y el ViT; en la ResNet solo con rewinding.
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
