import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final: Lottery Ticket Hypothesis a través de Arquitecturas
    Este Notebook es parte del Proyecto Final del curso de Deep Learning.

    ## 05 - Comparación entre arquitecturas
    Reúne los resultados guardados por los notebooks 01-04 (`results/*.json`) para responder las dos preguntas del proyecto:

    1. **¿Existen winning tickets en todas las arquitecturas?** Es decir, ¿el ticket supera consistentemente a la reinicialización aleatoria con la misma estructura?
    2. **¿Cuándo se "fija" un ticket?** ¿Regresar a la inicialización basta, o hace falta rebobinar a un punto posterior del entrenamiento? ¿Depende eso de la arquitectura?

    > Este notebook no entrena nada: primero hay que correr los notebooks 01-04. Las arquitecturas cuyo archivo de resultados no exista se omiten.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    # (archivo, nombre) de cada experimento
    EXPERIMENTS = {
        "mlp_mnist": "MLP / MNIST",
        "cnn_cifar10": "CNN / CIFAR-10",
        "resnet18_cifar10": "ResNet-18 / CIFAR-10",
        "vit_cifar10": "ViT / CIFAR-10",
    }

    runs = {}
    for _file, _name in EXPERIMENTS.items():
        _path = Path("../results") / f"{_file}.json"
        if _path.exists():
            runs[_name] = json.loads(_path.read_text())

    print(f"Resultados encontrados: {list(runs)}")
    return mo, plt, runs


@app.cell
def _(mo, runs):
    mo.stop(not runs, mo.md("**No hay resultados todavía.** Corre los notebooks 01-04 primero."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curvas por arquitectura
    Accuracy de prueba contra el porcentaje de pesos restantes (eje x logarítmico e invertido: la red se hace más pequeña hacia la derecha). En las arquitecturas con rewinding tardío (ResNet-18 y ViT) se muestra también esa curva.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(plt, runs):
    _labels = {
        "ticket": "Ticket (rewind a época 0)",
        "ticket_late": "Ticket (rewind tardío)",
        "random": "Reinit aleatorio",
    }
    _fig, _axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4), squeeze=False)
    for _ax, (_name, _run) in zip(_axes[0], runs.items()):
        for _key, _label in _labels.items():
            if _key in _run:
                _res = _run[_key]
                _ax.plot([r["weights_left"] * 100 for r in _res], [r["acc"] for r in _res], marker="o", markersize=4, label=_label)
        _ax.set_xscale("log")
        _ax.invert_xaxis()
        _ax.set_title(_name)
        _ax.set_xlabel("Pesos restantes (%)")
        _ax.grid(alpha=0.3)
    _axes[0][0].set_ylabel("Accuracy de prueba")
    _axes[0][0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy relativa a la red densa
    Para poder comparar arquitecturas con accuracies base muy distintas, restamos la accuracy de la red densa (ronda 0) de cada curva. Un valor de 0 significa que la subred iguala a la red completa.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(plt, runs):
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    for _i, (_name, _run) in enumerate(runs.items()):
        _color = f"C{_i}"
        _dense = _run["ticket"][0]["acc"]
        _ax.plot(
            [r["weights_left"] * 100 for r in _run["ticket"]],
            [r["acc"] - _dense for r in _run["ticket"]],
            marker="o", markersize=4, color=_color, label=f"{_name}: ticket",
        )
        _ax.plot(
            [r["weights_left"] * 100 for r in _run["random"]],
            [r["acc"] - _dense for r in _run["random"]],
            linestyle="--", color=_color, alpha=0.6, label=f"{_name}: aleatorio",
        )
    _ax.axhline(0, color="gray", linewidth=0.8)
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Pesos restantes (%)")
    _ax.set_ylabel("Δ accuracy vs. red densa")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=7, ncol=2)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resumen
    Para cada arquitectura se reporta:

    - **Mín. pesos (ticket / aleatorio):** la red más pequeña que queda a menos de 1 punto de la accuracy densa.
    - **Ventaja del ticket:** diferencia promedio de accuracy entre el ticket y la reinicialización aleatoria sobre las rondas podadas (mayor = la inicialización importa más).
    - **Ganancia del rewind tardío:** diferencia promedio entre el ticket con rewind tardío y el clásico (solo ResNet-18 y ViT). Si es positiva, el ticket **no** está fijo en $\theta_0$ y se fija más adelante en el entrenamiento.
    """)
    return


@app.cell
def _(mo, runs):
    def _smallest(res, dense):
        _ok = [r["weights_left"] for r in res if r["acc"] >= dense - 0.01]
        return f"{min(_ok):.1%}" if _ok else "ninguna"

    def _mean_gap(a, b):
        # Promedio de a - b sobre rondas podadas (sin la ronda 0)
        return sum(x["acc"] - y["acc"] for x, y in zip(a[1:], b[1:])) / len(a[1:])

    _rows = [
        "| Arquitectura | Acc. densa | Mín. pesos (ticket) | Mín. pesos (aleatorio) | Ventaja del ticket | Ganancia del rewind tardío |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _name, _run in runs.items():
        _dense = _run["ticket"][0]["acc"]
        _adv = _mean_gap(_run["ticket"], _run["random"])
        _late = f"{_mean_gap(_run['ticket_late'], _run['ticket']):+.2%}" if "ticket_late" in _run else "—"
        _rows.append(
            f"| {_name} | {_dense:.2%} | {_smallest(_run['ticket'], _dense)} "
            f"| {_smallest(_run['random'], _dense)} | {_adv:+.2%} | {_late} |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discusión
    Qué buscar en las gráficas y la tabla:

    - **Ticket vs. aleatorio.** Si la ventaja es claramente positiva, los valores iniciales de los pesos sobrevivientes importan (no solo la estructura). Se espera que crezca con la sparsidad.
    - **Rewind tardío.** En redes pequeñas y poco profundas (MLP, CNN) el ticket se encuentra en $\theta_0$. En redes más profundas o con optimizadores sensibles (ResNet-18, ViT), Frankle et al. (2019) [2] reportan que el ticket solo aparece si se rebobina a una época temprana. Para ViT, [5] reporta que los tickets son difíciles de encontrar incluso con rewinding, así que una ganancia nula allí no sería sorprendente. Una ganancia positiva en esas arquitecturas apoya la idea de que lo que se "fija" durante las primeras épocas es lo que hace estable al ticket, y no la arquitectura en sí.
    - **Limitaciones.** Una sola semilla por experimento, pocas épocas por ronda en ResNet-18 y ViT, y un solo valor de rewind. Las diferencias de menos de ~0.5 puntos no son concluyentes; para sacar conclusiones firmes habría que repetir con varias semillas y barrer $k$.
    """)
    return


if __name__ == "__main__":
    app.run()
