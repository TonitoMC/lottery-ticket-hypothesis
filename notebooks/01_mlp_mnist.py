import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es el punto de partida para el Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    > Una red densa, inicializada aleatoriamente, contiene una subred (un "winning ticket") que, entrenada de forma aislada **desde su misma inicialización**, alcanza una accuracy comparable a la de la red completa.

    Es decir, la mayoría de los pesos son prescindibles, pero importa *cuáles* se conservan **y** con qué valores empiezan.

    ## 01 - MNIST MLP
    Este Notebook sirve como un pequeño "proof of concept", donde se implementa el experimento detallado en Frankle & Carbin sobre una variación de LeNet. Se busca establecer una estructura general para los demás notebooks y arquitecturas más complejas, al igual que generar un pequeño baseline para las partes comparativas del proyecto.

    El código de pruning (poda por magnitud e IMP) está en `lth.py` para reutilizarlo en los demás notebooks.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    from torch.utils.data import Dataset
    from torch import nn
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    import lth

    # Set GPU - CPU como device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return datasets, device, lth, mo, nn, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    Para este experimento estaremos utilizando el dataset de MNIST (60,000 imágenes de entrenamiento y 10,000 de prueba de dígitos escritos a mano, 28x28 en escala de grises), elegido por ser un dataset clásico y además fácil de acceder con los datasets de torch.

    Empezamos cargando los conjuntos de prueba y entrenamiento, asegurándonos que las imágenes sean cargadas como tensores para cada uno.
    """)
    return


@app.cell
def _(datasets, torch, v2):
    # Datos de entrenamiento
    mnist_train = datasets.MNIST(
        root="data",
        train=True,
        # Descargar si no está presente en el directorio "data"
        download=True,
        # Imagen PIL -> Tensor -> Tensor de Float32
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    # Datos de prueba
    mnist_test = datasets.MNIST(
        root="data",
        train=False,
        # Descargar si no está presente en el directorio "data"
        download=True,
        # Imagen PIL -> Tensor -> Tensor de Float32
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )
    return mnist_test, mnist_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Muestra del Conjunto de Entrenamiento
    Revisamos un ejemplo de cada dígito (0-9) para confirmar que los datos y sus etiquetas se cargaron bien.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(mnist_train, plt):
    _fig, _axes = plt.subplots(2, 5, figsize=(9, 4))
    for _c, _ax in enumerate(_axes.flat):
        # Índice de la primera imagen de la clase _c
        _i = (mnist_train.targets == _c).nonzero()[0].item()
        _ax.imshow(mnist_train[_i][0].squeeze(), cmap="gray")
        _ax.set_title(f"Clase {_c}")
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loader
    Usamos un batch de 60, igual que el paper original de LTH. Revisamos la forma de un batch.
    """)
    return


@app.cell
def _(mnist_test, mnist_train):
    from torch.utils.data import DataLoader

    # Batch size de 60 (paper original de LTH)
    train_loader = DataLoader(mnist_train, 60, shuffle=True)
    test_loader = DataLoader(mnist_test, 60, shuffle=False)

    x, y = next(iter(train_loader))
    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP
    Este MLP es LeNet-300-100, la misma arquitectura del paper original. Sigue la siguiente arquitectura:
    - Input: 28x28 = 784
    - Capa Densa: 784 -> 300 -> ReLU
    - Capa Densa: 300 -> 100 -> ReLU
    - Capa Densa: 100 -> 10

    El modelo regresa **logits** (sin softmax): `CrossEntropyLoss` ya aplica el softmax internamente.
    """)
    return


@app.cell
def _(device, nn):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()

            # Aplanar de 28x28 a 784
            self.flatten = nn.Flatten()

            # Capas densas
            self.layers = nn.Sequential(
                nn.Linear(784, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
            )

        # Forward: Aplanar y pasar por capas densas
        def forward(self, x):
            x = self.flatten(x)
            logits = self.layers(x)
            # Retornar logits
            return logits

    # Instanciar modelo
    model = MLP()

    # Pasar a GPU
    model.to(device)

    # Inspeccionar
    model
    return MLP, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Podemos confirmar que el modelo fue creado exitosamente, por lo que podemos pasar a la parte de entrenamiento.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento
    Iniciamos primero almacenando una copia de los pesos iniciales antes del entrenamiento, porque el LTH necesita regresar a ellos después de podar.
    """)
    return


@app.cell
def _(model):
    import copy

    init_state = copy.deepcopy(model.state_dict())
    return (init_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Utilizamos entropía cruzada como función de pérdida y Adam como optimizador (con el learning rate del paper para LeNet, `1.2e-3`). No se usa regularización ni aumento de datos, para que el efecto del pruning sea lo único que cambie entre experimentos.
    """)
    return


@app.cell
def _(model, nn, torch):
    # Pérdida de clasificación multiclase
    loss_fn = nn.CrossEntropyLoss()

    # Adam con el learning rate del paper
    optimizer = torch.optim.Adam(model.parameters(), lr = 1.2e-3)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop de Entrenamiento
    Aquí definimos `train` (una época) y `evaluate` (accuracy en el conjunto de prueba), y corremos un pequeño loop de prueba para verificar que el MLP si logre realizar predicciones adecuadas.
    """)
    return


@app.cell
def _(device, loss_fn, model, optimizer, test_loader, torch, train_loader):
    # Función de una época de entrenamiento
    def train(model, loader, loss_fn, optimizer):
        # Colocar el modelo en modo entrenamiento
        model.train()
        # Para cada entrada del loader
        for x, y in loader:
            # Pasar a GPU
            x, y = x.to(device), y.to(device)
            # Logits del forward
            logits = model(x)
            # Calcular pérdida
            loss = loss_fn(logits, y)
            # Reiniciar gradiente
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Paso de optimizador
            optimizer.step()

    # Función de evaluación en conjunto de prueba
    def evaluate(model, loader):
        # Colocar el modelo en modo evaluación
        model.eval()
        # Predicciones correctas
        correct = 0
        # No almacenar gradiente
        with torch.no_grad():
            # Para cada entrada del loader
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # Predicción = mayor logit
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
        # Retorna % de correctos
        return correct / len(loader.dataset)

    # Loop de entrenamiento por 5 épocas
    for epoch in range(5):
        train(model, train_loader, loss_fn, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: test acc = {acc:.4f}")
    return (acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Podemos ver que se alcanza alrededor del 98% de accuracy, lo cuál es bastante bueno y alrededor de lo esperado utilizando una arquitectura simple. Este modelo entrenado es nuestro baseline (100% de los pesos).

    ## Pruning
    Esta sección corre el experimento de LTH, cuyo código está en `lth.py` para reutilizarlo en los demás notebooks. Buscamos mantener diferentes porcentajes de los pesos y comparar un entrenamiento desde los pesos iniciales originales vs desde pesos aleatorios nuevos. De esta manera identificar los "winning tickets".
    """)
    return


@app.cell
def _(model):
    for name, p in model.named_parameters():
        print(name, p.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Iterative Magnitude Pruning (IMP)
    Un solo corte apenas encuentra tickets pequeños, así que se repite en rondas. En cada ronda:

    1. Se poda el 20% de los pesos que quedan, por capa, los de menor magnitud $|w|$.
    2. Los pesos restantes regresan a su valor inicial (`init_state`).
    3. Se re-entrena la subred por 5 épocas.

    Tras $n$ rondas quedan $0.8^n$ de los pesos: 15 rondas dejan ~3.5%.

    Como control, se entrena la misma estructura pero con pesos aleatorios nuevos. Para saber si importan los *valores iniciales* y no solo la estructura: si la hipótesis es cierta, esta curva debe caer antes que la del winning ticket.
    """)
    return


@app.cell
def _(MLP, acc, device, init_state, lth, model, test_loader, train_loader):
    ticket_results, random_results = lth.run_lth(
        model, init_state, MLP, acc, train_loader, test_loader, device,
        rounds=15, epochs=5, lr=1.2e-3
    )
    return random_results, ticket_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resultados
    El eje x está en escala logarítmica e invertido: la red se hace más pequeña hacia la derecha.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(plt, random_results, ticket_results):
    _fig, _ax = plt.subplots()
    _ax.plot([r["weights_left"] * 100 for r in ticket_results],
             [r["acc"] for r in ticket_results], marker="o", label="Winning ticket")
    _ax.plot([r["weights_left"] * 100 for r in random_results],
             [r["acc"] for r in random_results], marker="o", label="Random reinit")
    _ax.set_xscale("log")
    _ax.invert_xaxis()
    _ax.set_xlabel("Weights remaining (%)")
    _ax.set_ylabel("Test accuracy")
    _ax.legend()
    _ax
    return


@app.cell
def _(random_results, ticket_results):
    # Guardar para el notebook de comparación (05)
    import json

    with open("../results/mlp_mnist.json", "w") as f:
        json.dump({"ticket": ticket_results, "random": random_results}, f)
    return


if __name__ == "__main__":
    app.run()
