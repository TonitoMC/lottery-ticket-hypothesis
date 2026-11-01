import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lottery Ticket Hypothesis a través de Arquitecturas
    **Proyecto Final - Deep Learning**

    > **Estado.** Todas las corridas terminaron (excepto `02b`, poda global de la CNN, que queda opcional). Las tablas y gráficas se calculan de `results/*.json`. Los bloques marcados **[POR COMPLETAR]** son para el análisis propio del autor; los textos descriptivos de las figuras solo dicen qué se ve, no por qué.

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
    **MLP / MNIST.** El ticket se mantiene entre 97.2% y 98.2% hasta 3.5% de los pesos, a la par de la red densa (97.7%), mientras que la reinicialización aleatoria cae a 95.7%. Regresar a la inicialización ya basta; no se probó el rewinding aquí. La máscara de la primera capa, dibujada como imagen de 28x28, poda sobre todo el borde (128 de los 784 pixeles de entrada quedan sin ningún peso), y en la segunda capa 54 de las 100 neuronas quedan sin pesos; es esperable porque los pixeles que siempre valen 0 nunca reciben gradiente.

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
    - El rewinding no cambia nada arriba de ~10% de los pesos; por debajo, mejora al ticket clásico (+3.5 puntos a 7.5%, +3.3 a 4.9%, +1.1 a 3.2%), en una sola corrida.
    - Por encima de la red densa: ticket clásico hasta 11.6%; con rewinding hasta 7.5%. [5] reporta que los tickets de peso son difíciles en ViTs; aquí el efecto puede deberse en parte a regularización por la poda, y con una semilla no se puede separar.

    **[POR COMPLETAR]** Interpretación propia: ¿por qué el rewinding solo ayuda en los puntos muy esparcidos? ¿Se puede relacionar con que 04 (20 épocas) no mostrara ventaja del rewinding?

    ## 6. Inspección de los tickets
    Checkpoints del ViT largo, comparados al mismo porcentaje de pesos (11.6%, ronda 5) con `06_inspection.py`. El código de las figuras fue generado con IA; los pesos y las máscaras son los de las corridas. Las descripciones solo dicen lo que se ve.
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
    **[POR COMPLETAR]** Interpretación propia de las figuras. Preguntas guía:

    - ¿Por qué `head` y `patch_embed` conservan más pesos que las capas MLP con poda global? ¿Qué dice eso de dónde están los pesos de mayor magnitud?
    - Si la atención del control aleatorio se concentra abajo, ¿es un efecto de la inicialización, de la máscara o de ambas? (misma máscara, otra inicialización)
    - ¿Es relevante que el rewinding empiece con gradientes menores? ¿Se relaciona con el warmup?
    - ¿La ventaja del ticket sobre el denso viene de unas pocas clases o de todas? (con ruido de ±2.5 por clase)

    La mismas vistas, con más detalle (cabezas de atención, IoU entre el ticket y el rewinding, forma de las máscaras, entropía de la atención), están en `06_inspection.html`.

    ## 7. Discusión
    **Qué sí apoyan los resultados.**
    - En el MLP y (hasta cierto punto) la CNN, regresar a la inicialización basta: el ticket iguala o supera a la red densa y le gana al control aleatorio.
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
    7. El colapso a 10.7% se repitió y coincide con filtros y neuronas completamente vacíos. ¿Sería evitable con poda global, que puede podar menos las capas pequeñas? *(Notebook 02b.)*
    8. ¿Una capa con 7 de 64 filtros vacíos es un problema de capacidad o solo desperdicio de pesos? *(Podar manteniendo un mínimo por filtro.)*

    **Sobre las máscaras.**
    9. ¿Por qué la atención del control aleatorio se concentra abajo si comparte la máscara con el ticket? ¿Qué papel juega la inicialización en dónde "mira" la red?
    10. ¿Qué tanto se parecen las máscaras del ticket y del rewinding (IoU, ver 06)? Si se parecen poco pero ambos funcionan, ¿hay muchos tickets distintos?

    **Sobre la metodología.**
    11. Con una sola semilla y ~10,000 imágenes de prueba, ¿qué diferencias son creíbles? (Un intervalo binomial del 95% sobre una accuracy de ~88% mide ±0.6 puntos; las diferencias entre ticket y aleatorio de 5-9 puntos lo superan, pero la de ticket contra denso a 7.5% no.)
    12. ¿Cómo afecta que la poda esté global en unos modelos y por capa en otros a la comparación entre arquitecturas?

    ## 9. Conclusiones y trabajo futuro
    1. Se encontraron winning tickets en el MLP, la CNN (con reserva en el punto más pequeño) y el ViT (también con 100 épocas y mejor baseline); en la ResNet solo con rewinding.
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
