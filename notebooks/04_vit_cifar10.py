import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    ## 04 - CIFAR-10 Vision Transformer
    Última arquitectura: un **ViT pequeño** entrenado desde cero en CIFAR-10. Los transformers son conocidos por ser **inestables al inicio del entrenamiento** (de ahí el warmup del learning rate), así que son un buen caso para la pregunta del proyecto: si esa inestabilidad temprana es lo que impide que un ticket se fije en la inicialización, el ViT podría beneficiarse del *rewinding* [2].

    Ojo: [5] reporta que en DeiT-Small (ImageNet) los winning tickets son difíciles de encontrar incluso con rewinding. Aquí es un ViT mucho más pequeño en CIFAR-10, así que no se puede asumir el mismo resultado, pero ninguna curva de ticket superando a la aleatoria también sería un resultado válido. Las referencias numeradas están en `REFERENCES.md`.

    Como en el notebook 03 se compara regresar a la inicialización, regresar a la época 2 (rewinding) y la misma estructura con pesos aleatorios.

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import copy
    import math
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
    return F, copy, datasets, device, lth, math, mo, nn, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    CIFAR-10 con aumento de datos en entrenamiento (volteo horizontal y recorte aleatorio con padding de 4) y normalización por canal. Los transformers entrenados desde cero en datasets pequeños dependen mucho del aumento.

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
    Usamos un batch de 128. Revisamos la forma de un batch.
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
    ## Vision Transformer
    Se implementa desde cero para controlar qué pesos se podan:

    1. **Patch embedding:** la imagen de 32x32 se divide en 64 parches de 4x4 y cada uno se proyecta a un vector de 192 dimensiones (una `Conv2d` con kernel y stride 4).
    2. Se antepone un token `[CLS]` y se suman **embeddings posicionales** aprendidos.
    3. **6 bloques transformer** (pre-norm), cada uno con:
       - Self-attention de 3 cabezas: `qkv` (192 -> 576) y `proj` (192 -> 192).
       - MLP: 192 -> 384 -> 192 con GELU.
       - Conexiones residuales y LayerNorm.
    4. Una capa lineal sobre el token `[CLS]` da los 10 logits.

    **Qué se poda:** todas las capas `Linear` y la `Conv2d` del patch embedding. LayerNorm, biases, el token `[CLS]` y los embeddings posicionales no se podan.
    """)
    return


@app.cell
def _(F, device, nn, torch):
    class Block(nn.Module):
        def __init__(self, dim, heads, mlp_dim):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

        def forward(self, x):
            B, N, D = x.shape
            # [B, N, 3D] -> 3 x [B, heads, N, D/heads]
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

        def forward(self, x):
            # [B, 3, 32, 32] -> [B, 64, dim]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
            x = self.norm(self.blocks(x))
            return self.head(x[:, 0])

    # Instanciar modelo
    model = ViT()

    # Pasar a GPU
    model.to(device)

    # Inspeccionar
    model
    return ViT, model


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
def _(copy, model):
    init_state = copy.deepcopy(model.state_dict())
    return (init_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Utilizamos entropía cruzada como función de pérdida y AdamW como optimizador. Se sigue la receta de DeiT [4]: weight decay `0.05`, warmup lineal y decaimiento coseno. DeiT usa 5 épocas de warmup en ImageNet; aquí son 2 en un total de 20. El learning rate máximo (`1e-3`) **no** viene de DeiT (su regla daría `1.25e-4` con batch 128): es una elección sin ajustar.
    """)
    return


@app.cell
def _(math, model, nn, torch):
    # Pérdida de clasificación multiclase
    loss_fn = nn.CrossEntropyLoss()

    # AdamW con la receta de DeiT
    def make_opt(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    optimizer = make_opt(model)

    # Warmup lineal por 2 épocas y luego coseno hasta 0 (20 épocas en total)
    def lr_at(epoch):
        if epoch < 2:
            return 1e-3 * (epoch + 1) / 2
        return 0.5e-3 * (1 + math.cos(math.pi * (epoch - 2) / 18))

    return loss_fn, lr_at, make_opt, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop de Entrenamiento
    Aquí definimos `train` (una época) y `evaluate` (accuracy en el conjunto de prueba), y corremos el entrenamiento denso completo (20 épocas) con el schedule del lr. Al terminar la época 2 (justo al acabar el warmup) se guarda una copia de los pesos (`rewind_state`), que se usa más adelante para el *rewinding*.
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

    # Loop de entrenamiento por 20 épocas
    for epoch in range(20):
        # Learning rate del schedule
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch)
        train(model, train_loader, loss_fn, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: test acc = {acc:.4f}")
        # Pesos al terminar la época 2 (para el rewinding)
        if epoch == 1:
            rewind_state = copy.deepcopy(model.state_dict())
    return acc, rewind_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este modelo entrenado es nuestro baseline denso (100% de los pesos). Un ViT pequeño desde cero en CIFAR-10 queda por debajo de una ResNet; lo que importa aquí es cómo se degrada al podarlo.

    ## Pruning
    Esta sección corre el experimento de LTH, cuyo código está en `lth.py` para reutilizarlo en los demás notebooks, con poda **global** como en las redes profundas de [1].
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

    1. Se poda el 35% de los pesos que quedan, de menor magnitud $|w|$.
    2. Los pesos restantes regresan a su valor de referencia.
    3. Se re-entrena la subred con el schedule del lr.

    Tras $n$ rondas quedan $0.65^n$ de los pesos: 8 rondas dejan ~3%. Se comparan tres curvas:
    - **Ticket clásico**: regresa a la inicialización (`init_state`, 20 épocas).
    - **Ticket con rewinding**: regresa a los pesos tras la época 2 y re-entrena las 18 épocas restantes del schedule.
    - **Control**: la misma estructura del ticket clásico pero con pesos aleatorios nuevos.
    """)
    return


@app.cell
def _(ViT, acc, device, init_state, lr_at, lth, make_opt, model, rewind_state, test_loader, train_loader):
    ticket_results, late_results, random_results = lth.run_lth(
        model, init_state, ViT, acc, train_loader, test_loader, device,
        rounds=8, epochs=20, lr=1e-3, rate=0.35, scope="global",
        rewind_state=rewind_state, rewind_epoch=2, make_opt=make_opt, lr_at=lr_at
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
             [r["acc"] for r in late_results], marker="o", label="Ticket (rewind a época 2)")
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

    with open("../results/vit_cifar10.json", "w") as f:
        json.dump({"ticket": ticket_results, "ticket_late": late_results, "random": random_results}, f)
    return


if __name__ == "__main__":
    app.run()
