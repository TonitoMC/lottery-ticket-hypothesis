import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    > Una red densa, inicializada aleatoriamente, contiene una subred (un "winning ticket") que, entrenada de forma aislada **desde su misma inicialización**, alcanza una accuracy comparable a la de la red completa.

    Es decir, la mayoría de los pesos son prescindibles, pero importa *cuáles* se conservan **y** con qué valores empiezan.

    ## 02 - CIFAR-10 CNN
    En el notebook 01 se reprodujo el resultado de LTH con un MLP en MNIST. Este Notebook repite el experimento con una **CNN** (Conv-4 del paper de Frankle & Carbin) en **CIFAR-10**, un problema más difícil donde los pesos tienen estructura espacial (filtros convolucionales). Se sigue la misma estructura del notebook 01, al igual que generar otro baseline para las partes comparativas del proyecto.

    El código de pruning (poda por magnitud e IMP) está en `lth.py` y es el mismo que usa el notebook 01.

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
    CIFAR-10 tiene 50,000 imágenes de entrenamiento y 10,000 de prueba, a color de 32x32 y en 10 clases (avión, auto, pájaro, gato, venado, perro, rana, caballo, barco y camión). Se cargan como tensores de Float32 en $[0, 1]$, elegido por ser un dataset clásico y además fácil de acceder con los datasets de torch.

    Empezamos cargando los conjuntos de prueba y entrenamiento, asegurándonos que las imágenes sean cargadas como tensores para cada uno.
    """)
    return


@app.cell
def _(datasets, torch, v2):
    cifar_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    cifar_test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )
    return cifar_test, cifar_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Muestra del Conjunto de Entrenamiento
    Revisamos un ejemplo de cada clase para confirmar que los datos y sus etiquetas se cargaron bien. Son imágenes de baja resolución (32x32), lo que hace al problema más difícil que MNIST.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(cifar_train, plt, torch):
    _targets = torch.tensor(cifar_train.targets)
    _fig, _axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for _c, _ax in enumerate(_axes.flat):
        # Índice de la primera imagen de la clase _c
        _i = (_targets == _c).nonzero()[0].item()
        # [C, H, W] -> [H, W, C] para matplotlib
        _ax.imshow(cifar_train[_i][0].permute(1, 2, 0))
        _ax.set_title(cifar_train.classes[_c])
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loader
    Usamos un batch de 64. Se usan varios workers para que la carga de imágenes no sea el cuello de botella. Revisamos la forma de un batch.
    """)
    return


@app.cell
def _(cifar_test, cifar_train):
    from torch.utils.data import DataLoader

    # Batch de 64, shuffle solo en entrenamiento
    train_loader = DataLoader(cifar_train, 64, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(cifar_test, 64, shuffle=False, num_workers=4, persistent_workers=True)

    x, y = next(iter(train_loader))

    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CNN
    Esta CNN es Conv-4, la misma arquitectura del paper original. Sigue la siguiente arquitectura:
    - Input: 3x32x32
    - Conv 3x3: 3 -> 64 -> ReLU
    - Conv 3x3: 64 -> 64 -> ReLU -> MaxPool (64x16x16)
    - Conv 3x3: 64 -> 128 -> ReLU
    - Conv 3x3: 128 -> 128 -> ReLU -> MaxPool (128x8x8 = 8192)
    - Capa Densa: 8192 -> 256 -> ReLU
    - Capa Densa: 256 -> 256 -> ReLU
    - Capa Densa: 256 -> 10

    Las convoluciones usan `padding=1`, por lo que conservan el tamaño espacial y es el `MaxPool` quien lo reduce a la mitad. Casi todos los pesos (~2.1M de ~2.4M) están en la primera capa densa.

    El modelo regresa **logits** (sin softmax): `CrossEntropyLoss` ya aplica el softmax internamente.
    """)
    return


@app.cell
def _(device, nn):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                # [3, 32, 32] -> [64, 32, 32]
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                # [64, 32, 32] -> [64, 32, 32]
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size = 3, padding=1),
                nn.ReLU(),
                # [64, 32, 32] -> [64, 16, 16]
                nn.MaxPool2d(kernel_size=2),

                # [64, 16, 16] -> [128, 16, 16]
                nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, padding = 1),
                nn.ReLU(),
                # [128, 16, 16] -> [128, 16, 16]
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, padding=1), nn.ReLU(),
                # [128, 16, 16] -> [128, 8, 8]
                nn.MaxPool2d(kernel_size = 2)
            )
            self.linear = nn.Sequential(
                # [128, 8, 8] -> 8192
                nn.Flatten(),
                nn.Linear(8192, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256,10)
            )
        def forward(self, x):
            return self.linear(self.conv(x))

    cnn = CNN()

    cnn.to(device)

    cnn
    return CNN, cnn


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
    Igual que en el notebook 01: iniciamos almacenando una copia de los pesos iniciales antes del entrenamiento, porque el LTH necesita regresar a ellos después de podar.
    """)
    return


@app.cell
def _(cnn):
    import copy

    init_state = copy.deepcopy(cnn.state_dict())
    return (init_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Utilizamos entropía cruzada como función de pérdida y Adam como optimizador (con el learning rate del paper para Conv-4, `3e-4`). No se usa regularización ni aumento de datos, para que el efecto del pruning sea lo único que cambie entre experimentos.
    """)
    return


@app.cell
def _(cnn, nn, torch):
    # Pérdida de clasificación multiclase
    loss_fn = nn.CrossEntropyLoss()

    # Adam con el learning rate del paper
    optimizer = torch.optim.Adam(cnn.parameters(), lr = 3e-4)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop de Entrenamiento
    Aquí definimos `train` (una época) y `evaluate` (accuracy en el conjunto de prueba), y corremos un pequeño loop de prueba para verificar que la CNN si logre realizar predicciones adecuadas. Se entrena por 30 épocas, que son ~25K iteraciones como en el paper.
    """)
    return


@app.cell
def _(cnn, device, loss_fn, optimizer, test_loader, torch, train_loader):
    # Una época de entrenamiento
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

    # Accuracy en el conjunto de prueba
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

    # Loop de entrenamiento por 30 épocas
    for epoch in range(30):
        train(cnn, train_loader, loss_fn, optimizer)
        acc = evaluate(cnn, test_loader)
        print(f"Epoch {epoch + 1}: test acc = {acc:.4f}")
    return (acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este modelo entrenado es el baseline (100% de los pesos). Sin aumento de datos ni regularización se sobreajusta, así que la accuracy se queda por debajo de lo que lograría una CNN más regularizada.

    ## Pruning
    Esta sección corre el experimento de LTH, cuyo código está en `lth.py` para reutilizarlo en los demás notebooks. Buscamos mantener diferentes porcentajes de los pesos y comparar un entrenamiento desde los pesos iniciales originales vs desde pesos aleatorios nuevos. De esta manera identificar los "winning tickets".

    ### Iterative Magnitude Pruning (IMP)
    Un solo corte apenas encuentra tickets pequeños, así que se repite en rondas. En cada ronda:

    1. Se poda el 20% de los pesos que quedan, por capa (filtros convolucionales y capas densas), los de menor magnitud $|w|$.
    2. Los pesos restantes regresan a su valor inicial (`init_state`).
    3. Se re-entrena la subred por 30 épocas.

    Tras $n$ rondas quedan $0.8^n$ de los pesos: 10 rondas dejan ~10.7%. El paper poda las convoluciones más despacio (10%) que las capas densas (20%); aquí se usa 20% en todas por simplicidad.

    Como control, se entrena la misma estructura pero con pesos aleatorios nuevos. Para saber si importan los *valores iniciales* y no solo la estructura: si la hipótesis es cierta, esta curva debe caer antes que la del winning ticket.

    Nota: el paper reporta que los winning tickets de las CNN pueden incluso *mejorar* la accuracy del baseline al podar.
    """)
    return


@app.cell
def _(cnn):
    for name, p in cnn.named_parameters():
        print(name, p.shape)
    return


@app.cell
def _(CNN, acc, cnn, device, init_state, lth, test_loader, train_loader):
    ticket_results, random_results = lth.run_lth(
        cnn, init_state, CNN, acc, train_loader, test_loader, device,
        rounds=10, epochs=30, lr=3e-4
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

    with open("../results/cnn_cifar10.json", "w") as f:
        json.dump({"ticket": ticket_results, "random": random_results}, f)
    return


if __name__ == "__main__":
    app.run()
