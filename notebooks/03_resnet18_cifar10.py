import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    ## 03 - CIFAR-10 ResNet-18
    En los notebooks anteriores el winning ticket se encontraba regresando los pesos a su **inicialización**. En redes más profundas eso falla: Frankle & Carbin [1] no encuentran tickets en su ResNet-18 con `lr = 0.1`, y Frankle et al. [2] muestran que sí aparecen si en vez de regresar a la inicialización se regresa a un punto **temprano del entrenamiento** (~500 iteraciones). A esto se le llama *rewinding*.

    Este notebook usa la misma red y receta del paper, y compara:
    - Regresar a la inicialización (ticket clásico).
    - Regresar a la época 1 del entrenamiento (rewinding).
    - La misma estructura con pesos aleatorios (control).

    Es el núcleo de la pregunta del proyecto: *¿cuándo se "fija" un ticket?* Las referencias numeradas están en `REFERENCES.md`.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import copy
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    import lth

    # Set GPU - CPU as device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return F, copy, datasets, device, lth, mo, nn, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    CIFAR-10 (50,000 imágenes de entrenamiento y 10,000 de prueba, 32x32 a color). Como en el paper, el conjunto de entrenamiento lleva **aumento de datos**: volteo horizontal aleatorio y recorte aleatorio con padding de 4 pixeles. Ambos conjuntos se normalizan por canal.

    Empezamos cargando los conjuntos de prueba y entrenamiento, asegurándonos que las imágenes sean cargadas como tensores para cada uno.
    """)
    return


@app.cell
def _(datasets, torch, v2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]

    cifar_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] + to_tensor)
    )

    cifar_test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform = v2.Compose(to_tensor)
    )

    return cifar_test, cifar_train, mean, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Muestra del Conjunto de Entrenamiento
    Revisamos un ejemplo de cada clase para confirmar que los datos y sus etiquetas se cargaron bien. Se deshace la normalización para poder verlas (con el aumento de datos aplicado).

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(cifar_train, mean, plt, std, torch):
    _targets = torch.tensor(cifar_train.targets)
    _mean, _std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    _fig, _axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for _c, _ax in enumerate(_axes.flat):
        # Índice de la primera imagen de la clase _c
        _i = (_targets == _c).nonzero()[0].item()
        # Deshacer la normalización para visualizar
        _img = cifar_train[_i][0] * _std + _mean
        # [C, H, W] -> [H, W, C] para matplotlib
        _ax.imshow(_img.permute(1, 2, 0).clamp(0, 1))
        _ax.set_title(cifar_train.classes[_c])
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loader
    Usamos un batch de 128, como en el paper. Revisamos la forma de un batch.
    """)
    return


@app.cell
def _(cifar_test, cifar_train):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(cifar_train, 128, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(cifar_test, 128, shuffle=False, num_workers=4, persistent_workers=True)

    x, y = next(iter(train_loader))
    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ResNet-18 (la del paper)
    La "Resnet-18" de [1] **no** es la ResNet-18 de ImageNet (11M de parámetros): es la ResNet para CIFAR de He et al. con **20 capas** y ~274K parámetros. Se construye a mano:

    - **Entrada:** `Conv 3x3` (3 -> 16) + BatchNorm + ReLU.
    - **Tres etapas** de 3 bloques residuales con 16, 32 y 64 canales. Las etapas 2 y 3 empiezan con stride 2 (32x32 -> 16x16 -> 8x8).
    - **Bloque residual:** `Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN`, se suma la entrada y luego ReLU. Si cambian las dimensiones, el atajo es una `Conv 1x1` con stride + BN.
    - **Salida:** average pooling global + capa lineal (64 -> 10).

    Son 1 + 9x2 = 19 convoluciones 3x3 más la capa lineal: 20 capas. Los pesos se inicializan con Glorot gaussiano, como en el paper.
    """)
    return


@app.cell
def _(F, device, nn):
    class Block(nn.Module):
        def __init__(self, c_in, c_out, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(c_out)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)

            # Atajo: identidad, o conv 1x1 si cambian las dimensiones
            self.shortcut = nn.Identity()
            if stride != 1 or c_in != c_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + self.shortcut(x))

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)

            # 3 etapas de 3 bloques; la primera de las etapas 2 y 3 baja la resolución
            blocks, c_in = [], 16
            for i, c_out in enumerate([16, 32, 64]):
                for j in range(3):
                    blocks.append(Block(c_in, c_out, 2 if (j == 0 and i > 0) else 1))
                    c_in = c_out
            self.blocks = nn.Sequential(*blocks)
            self.fc = nn.Linear(64, 10)

            # Glorot gaussiano
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_normal_(m.weight)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.blocks(x)
            return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    # Instanciar modelo
    model = ResNet()

    # Pasar a GPU
    model.to(device)

    # Inspeccionar
    model
    return ResNet, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verificamos contra la Figura 2 de [1], que reporta 270K pesos convolucionales y 274K parámetros en total (contando los buffers de BatchNorm).
    """)
    return


@app.cell
def _(model, mo):
    conv_weights = sum(p.numel() for p in model.parameters() if p.dim() == 4)
    total = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for n, b in model.named_buffers() if "num_batches" not in n
    )
    mo.md(f"Pesos convolucionales: **{conv_weights:,}** | Parámetros + buffers de BN: **{total:,}**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento
    Iniciamos primero almacenando una copia de los pesos iniciales antes del entrenamiento, porque el LTH necesita regresar a ellos después de podar.
    """)
    return


@app.cell
def _(copy, model):
    init_state = copy.deepcopy(model.state_dict())
    return (init_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Utilizamos entropía cruzada como función de pérdida y SGD como optimizador, con la receta del paper (Apéndice I.3 de [1]): momentum 0.9, weight decay `1e-4`, `lr = 0.1` y 30K iteraciones, dividiendo el lr entre 10 a las 20K y 25K. Con 391 iteraciones por época son **77 épocas**, con los cortes en las épocas **51** y **64**.
    """)
    return


@app.cell
def _(model, nn, torch):
    # Pérdida de clasificación multiclase
    loss_fn = nn.CrossEntropyLoss()

    # SGD con la receta del paper
    def make_opt(model):
        return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    optimizer = make_opt(model)

    # lr = 0.1, dividido entre 10 en las épocas 51 y 64
    def lr_at(epoch):
        return 0.1 * 0.1 ** sum(epoch >= m for m in (51, 64))

    return loss_fn, lr_at, make_opt, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop de Entrenamiento
    Aquí definimos `train` (una época) y `evaluate` (accuracy en el conjunto de prueba), y corremos el entrenamiento denso completo (77 épocas) con el schedule del lr. Al terminar la época 1 se guarda una copia de los pesos (`rewind_state`), que se usa más adelante para el *rewinding*.
    """)
    return


@app.cell
def _(copy, device, loss_fn, lr_at, model, optimizer, test_loader, torch, train_loader):
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

    # Loop de entrenamiento por 77 épocas
    for epoch in range(77):
        # Learning rate del schedule
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch)
        train(model, train_loader, loss_fn, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: test acc = {acc:.4f}")
        # Pesos al terminar la época 1 (para el rewinding)
        if epoch == 0:
            rewind_state = copy.deepcopy(model.state_dict())
    return acc, rewind_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este modelo entrenado es nuestro baseline denso (100% de los pesos). El paper [1] reporta 90.5% para este setup.

    ## Pruning
    Esta sección corre el experimento de LTH, cuyo código está en `lth.py` para reutilizarlo en los demás notebooks, con dos diferencias que usa el paper para esta red:
    - Poda **global** (un solo umbral para toda la red): las capas tienen tamaños muy distintos, y podarlas todas igual convierte a las pequeñas en cuellos de botella.
    - Solo se podan las convoluciones: la capa lineal final (640 pesos) se deja intacta (`skip=("fc",)`).
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

    1. Se poda el 20% de los pesos que quedan, de menor magnitud $|w|$.
    2. Los pesos restantes regresan a su valor de referencia.
    3. Se re-entrena la subred con el schedule del lr.

    Tras $n$ rondas quedan $0.8^n$ de los pesos: 10 rondas dejan ~10.7%. Se comparan tres curvas:
    - **Ticket clásico**: regresa a la inicialización (`init_state`, 77 épocas). Según [1] y [2], con `lr = 0.1` este ticket **no** debería igualar a la red densa.
    - **Ticket con rewinding**: regresa a los pesos tras la época 1 (391 iteraciones; [2] usa ~500) y re-entrena las 76 épocas restantes del schedule.
    - **Control**: la misma estructura del ticket clásico pero con pesos aleatorios nuevos.
    """)
    return


@app.cell
def _(ResNet, acc, device, init_state, lr_at, lth, make_opt, model, rewind_state, test_loader, train_loader):
    ticket_results, late_results, random_results = lth.run_lth(
        model, init_state, ResNet, acc, train_loader, test_loader, device,
        rounds=10, epochs=77, lr=0.1, scope="global", skip=("fc",),
        rewind_state=rewind_state, rewind_epoch=1, make_opt=make_opt, lr_at=lr_at
    )
    return late_results, random_results, ticket_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resultados
    El eje x está en escala logarítmica e invertido: la red se hace más pequeña hacia la derecha.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(late_results, plt, random_results, ticket_results):
    _fig, _ax = plt.subplots()
    _ax.plot([r["weights_left"] * 100 for r in ticket_results],
             [r["acc"] for r in ticket_results], marker="o", label="Ticket (rewind a init)")
    _ax.plot([r["weights_left"] * 100 for r in late_results],
             [r["acc"] for r in late_results], marker="o", label="Ticket (rewind a época 1)")
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
def _(late_results, random_results, ticket_results):
    # Guardar para el notebook de comparación (05)
    import json

    with open("../results/resnet18_cifar10.json", "w") as f:
        json.dump({"ticket": ticket_results, "ticket_late": late_results, "random": random_results}, f)
    return


if __name__ == "__main__":
    app.run()
