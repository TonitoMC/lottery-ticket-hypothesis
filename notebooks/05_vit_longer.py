import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis" (LTH).

    ## 05 - ViT con entrenamiento más largo y más aumento de datos
    En el notebook 04 el ViT denso llegó a 78.9% en 20 épocas, probablemente **subentrenado** (ver el informe), y aun así el ticket podado quedó por encima del denso (82-83%). Antes de interpretar eso como "podar regulariza" hay que arreglar el baseline. Este notebook:

    1. Entrena el ViT denso mucho más tiempo (`EPOCHS = 100`), con aumento de datos agresivo (RandAugment fuerte, random erasing, mixup / cutmix) y label smoothing, y guarda **accuracy de entrenamiento vs prueba por época** para ver si hay sobreajuste o subentrenamiento.
    2. Solo si el baseline se ve bien, corre IMP completo (`RUN_IMP = True`). Es caro: son 26 entrenamientos de `EPOCHS` épocas.

    Todo lo demás (arquitectura, poda global del 35% por ronda, 8 rondas, rewinding) es igual que en 04 para que sean comparables. Las referencias numeradas están en la bibliografía del informe (`report/informe.tex`).

    ## Imports - Setup
    """)
    return


@app.cell
def _():
    import copy
    import json
    import math
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    import lth

    # Set GPU - CPU as device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return DataLoader, F, Subset, copy, datasets, device, json, lth, math, mo, nn, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuración
    - `EPOCHS`: épocas del entrenamiento denso y de cada re-entrenamiento de IMP (04 usaba 20).
    - `WARMUP`: épocas de warmup lineal. DeiT [4] usa 5.
    - `MIXUP_CUTMIX`: activa mixup / cutmix por batch.
    - `REWIND_EPOCH`: a qué época se regresa en el ticket con rewinding. Igual al final del warmup, como en 04.
    - `LABEL_SMOOTHING`: suavizado de etiquetas en la pérdida de entrenamiento (DeiT usa 0.1).
    - `RUN_IMP`: ponerlo en `True` hasta revisar el baseline denso.

    Tiempo aproximado de IMP: 04 tardó ~45 min con 20 épocas, así que `EPOCHS = 100` serían ~3h45m (sin contar el costo extra de la aumentación).
    """)
    return


@app.cell
def _():
    EPOCHS = 100
    WARMUP = 5
    REWIND_EPOCH = 5
    LABEL_SMOOTHING = 0.1
    MIXUP_CUTMIX = True
    RUN_IMP = True
    return EPOCHS, LABEL_SMOOTHING, MIXUP_CUTMIX, REWIND_EPOCH, RUN_IMP, WARMUP


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    CIFAR-10 con aumento de datos agresivo, siguiendo la idea de DeiT [4] (que usa RandAugment, random erasing, mixup y cutmix):
    - Por imagen: recorte aleatorio con padding de 4, volteo horizontal, **RandAugment** (2 operaciones, magnitud 14), normalización por canal y **random erasing** (p = 0.25).
    - Por batch (más abajo): **mixup o cutmix** (uno al azar), que producen etiquetas suaves. Se pasan a `lth.train_model` con `batch_tf`.

    Todo esto es nuevo respecto a 04, que solo tenía volteo y recorte.

    Además se crea una copia del conjunto de entrenamiento **sin aumento**, para medir la accuracy de entrenamiento del modelo denso en las mismas condiciones que la de prueba.
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
        transform = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), v2.RandAugment(num_ops=2, magnitude=14)] + to_tensor + [v2.RandomErasing(p=0.25)])
    )

    # Mismo conjunto de entrenamiento sin aumento (para medir train acc)
    cifar_train_clean = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform = v2.Compose(to_tensor)
    )

    cifar_test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform = v2.Compose(to_tensor)
    )

    return cifar_test, cifar_train, cifar_train_clean, mean, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Muestra del Conjunto de Entrenamiento
    Un ejemplo de cada clase con el aumento aplicado, para confirmar que RandAugment no destruye las imágenes. Se deshace la normalización para poder verlas.

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
    Batch de 128. El `train_eval_loader` usa las primeras 10,000 imágenes de entrenamiento sin aumento, suficiente para estimar la accuracy de entrenamiento sin gastar mucho tiempo por época.
    """)
    return


@app.cell
def _(DataLoader, Subset, cifar_test, cifar_train, cifar_train_clean):
    train_loader = DataLoader(cifar_train, 128, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(cifar_test, 128, shuffle=False, num_workers=4, persistent_workers=True)
    train_eval_loader = DataLoader(Subset(cifar_train_clean, range(10_000)), 128, shuffle=False, num_workers=4)

    x, y = next(iter(train_loader))
    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_eval_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vision Transformer
    Misma arquitectura que en 04: parches de 4x4 (64 parches) proyectados a 192 dimensiones, token `[CLS]`, embeddings posicionales aprendidos, 6 bloques transformer pre-norm (3 cabezas, MLP 192 -> 384 -> 192 con GELU) y una capa lineal sobre `[CLS]`.

    **Qué se poda:** todas las capas `Linear` y la `Conv2d` del patch embedding. LayerNorm, biases, `[CLS]` y los embeddings posicionales no se podan.
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
    return ViT, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento denso
    Primero se guarda una copia de los pesos iniciales (el LTH regresa a ellos después de podar). Se usa AdamW con weight decay `0.05` y warmup lineal + coseno como en 04, pero con `WARMUP` y `EPOCHS` de la configuración. El learning rate máximo (`1e-3`) sigue siendo una elección sin ajustar (la regla de DeiT [4] daría `1.25e-4` con batch 128).
    """)
    return


@app.cell
def _(copy, model):
    init_state = copy.deepcopy(model.state_dict())
    return (init_state,)


@app.cell
def _(MIXUP_CUTMIX, v2):
    # Mixup o cutmix al azar por batch, devuelve etiquetas suaves [B, 10]
    _mix = v2.RandomChoice([v2.MixUp(alpha=0.8, num_classes=10), v2.CutMix(alpha=1.0, num_classes=10)])

    def _mixed(x, y):
        return _mix(x, y)

    batch_tf = _mixed if MIXUP_CUTMIX else None
    return (batch_tf,)


@app.cell
def _(EPOCHS, LABEL_SMOOTHING, WARMUP, math, model, nn, torch):
    # Pérdida con label smoothing (solo en entrenamiento; la accuracy no cambia)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # AdamW con la receta de DeiT
    def make_opt(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    optimizer = make_opt(model)

    # Warmup lineal por WARMUP épocas y luego coseno hasta 0
    def lr_at(epoch):
        if epoch < WARMUP:
            return 1e-3 * (epoch + 1) / WARMUP
        return 0.5e-3 * (1 + math.cos(math.pi * (epoch - WARMUP) / (EPOCHS - WARMUP)))

    return loss_fn, lr_at, make_opt, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop de Entrenamiento
    `train` hace una época, `evaluate` mide la accuracy en un loader. En cada época se guarda la accuracy de **entrenamiento** (sin aumento, 10,000 imágenes), la de **prueba** y la pérdida promedio de entrenamiento. Al terminar la época `REWIND_EPOCH` se guarda una copia de los pesos (`rewind_state`) para el rewinding.
    """)
    return


@app.cell
def _(EPOCHS, REWIND_EPOCH, batch_tf, copy, device, loss_fn, lr_at, model, optimizer, test_loader, torch, train_eval_loader, train_loader):
    # Función de una época de entrenamiento, regresa la pérdida promedio
    def train(model, loader, loss_fn, optimizer, batch_tf=None):
        # Colocar el modelo en modo entrenamiento
        model.train()
        total = 0.0
        # Para cada entrada del loader
        for x, y in loader:
            # Pasar a GPU
            x, y = x.to(device), y.to(device)
            # Mixup / cutmix (etiquetas suaves)
            if batch_tf:
                x, y = batch_tf(x, y)
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
            total += loss.item() * len(x)
        return total / len(loader.dataset)

    # Función de evaluación
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

    history = []
    for epoch in range(EPOCHS):
        # Learning rate del schedule
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch)
        loss = train(model, train_loader, loss_fn, optimizer, batch_tf)
        train_acc = evaluate(model, train_eval_loader)
        acc = evaluate(model, test_loader)
        history.append({"epoch": epoch + 1, "loss": loss, "train_acc": train_acc, "test_acc": acc})
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}, train acc = {train_acc:.4f}, test acc = {acc:.4f}")
        # Pesos al terminar la época REWIND_EPOCH (para el rewinding)
        if epoch + 1 == REWIND_EPOCH:
            rewind_state = copy.deepcopy(model.state_dict())
    return acc, history, rewind_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diagnóstico del baseline
    Si la accuracy de entrenamiento queda cerca de la de prueba y ambas siguen subiendo, el modelo está **subentrenado**; si entrenamiento llega a ~100% y prueba se estanca, está **sobreajustado**. Esto decide si tiene sentido interpretar "podar mejora" como regularización.

    *El código de las gráficas fue generado con IA.*
    """)
    return


@app.cell
def _(history, plt):
    _fig, _ax = plt.subplots()
    _ax.plot([h["epoch"] for h in history], [h["train_acc"] for h in history], label="Train (sin aumento)")
    _ax.plot([h["epoch"] for h in history], [h["test_acc"] for h in history], label="Test")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Accuracy")
    _ax.legend()
    _ax
    return


@app.cell
def _(acc, history, json):
    # Baseline denso para el informe (no pisa vit_cifar10.json, que lee 06_comparison)
    with open("../results/vit_longer_dense.json", "w") as _f:
        json.dump({"dense_acc": acc, "history": history}, _f)
    print(f"Dense final: test acc = {acc:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pruning
    Igual que en 04: IMP con poda **global** del 35% de los pesos que quedan por ronda, 8 rondas (~3% al final), con tres curvas:
    - **Ticket clásico**: regresa a la inicialización (`init_state`, `EPOCHS` épocas).
    - **Ticket con rewinding**: regresa a los pesos de la época `REWIND_EPOCH` y re-entrena el resto del schedule.
    - **Control**: la misma estructura del ticket clásico con pesos aleatorios nuevos.

    Cada ronda se guarda en `../checkpoints/05_vit_longer/` en cuanto termina (pesos + máscaras, con el % de pesos en el nombre). Esta celda no corre hasta poner `RUN_IMP = True`.
    """)
    return


@app.cell
def _(REWIND_EPOCH, acc, init_state, lth, model, rewind_state):
    # Guardar el modelo denso entrenado, sus pesos iniciales y los de rewind, aunque RUN_IMP sea False
    lth.save_base("../checkpoints/05_vit_longer", model, init_state, rewind_state, rewind_epoch=REWIND_EPOCH)
    return


@app.cell
def _(
    EPOCHS,
    LABEL_SMOOTHING,
    REWIND_EPOCH,
    RUN_IMP,
    batch_tf,
    ViT,
    acc,
    device,
    init_state,
    lr_at,
    lth,
    make_opt,
    mo,
    model,
    rewind_state,
    test_loader,
    train_loader,
):
    mo.stop(not RUN_IMP, mo.md("IMP desactivado: revisar el baseline y poner `RUN_IMP = True`."))
    ticket_results, late_results, random_results = lth.run_lth(
        model, init_state, ViT, acc, train_loader, test_loader, device,
        rounds=8, epochs=EPOCHS, lr=1e-3, rate=0.35, scope="global",
        rewind_state=rewind_state, rewind_epoch=REWIND_EPOCH, make_opt=make_opt, lr_at=lr_at,
        label_smoothing=LABEL_SMOOTHING, batch_tf=batch_tf, save_dir="../checkpoints/05_vit_longer"
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
def _(REWIND_EPOCH, late_results, plt, random_results, ticket_results):
    _fig, _ax = plt.subplots()
    _ax.plot([r["weights_left"] * 100 for r in ticket_results],
             [r["acc"] for r in ticket_results], marker="o", label="Ticket (rewind a init)")
    _ax.plot([r["weights_left"] * 100 for r in late_results],
             [r["acc"] for r in late_results], marker="o", label=f"Ticket (rewind a época {REWIND_EPOCH})")
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
def _(json, late_results, random_results, ticket_results):
    with open("../results/vit_longer.json", "w") as _f:
        json.dump({"ticket": ticket_results, "ticket_late": late_results, "random": random_results}, _f)
    return


if __name__ == "__main__":
    app.run()
