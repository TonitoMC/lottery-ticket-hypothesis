import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNIST MLP
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

    return datasets, mo, nn, plt, torch, v2


@app.cell
def _(torch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return (device,)


@app.cell
def _(datasets, torch, v2):
    # Training data
    mnist_train = datasets.MNIST(
        root="data",
        train=True,
        # Downloads if not present at data dir
        download=True,
        # PIL Image -> Tensor -> float32 tensor
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    mnist_test = datasets.MNIST(
        root="data",
        train=False,
        # Downloads if not present at data dir
        download=True,
        # PIL Image -> Tensor -> float32 tensor
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    return mnist_test, mnist_train


@app.cell
def _(mnist_train, plt):
    img, label = mnist_train[0]
    fig, ax = plt.subplots()
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(label)
    ax
    return


@app.cell
def _(mnist_test, mnist_train):
    from torch.utils.data import DataLoader

    # Batch size 60 (original LTH paper)
    train_loader = DataLoader(mnist_train, 60, shuffle=True)
    test_loader = DataLoader(mnist_test, 60, shuffle=False)

    x, y = next(iter(train_loader))
    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_loader


@app.cell
def _(device, nn):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layers = nn.Sequential(
                nn.Linear(784, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.layers(x)
            return logits

    model = MLP()

    model.to(device)


    model
    return (model,)


@app.cell(hide_code=True)
def _(model):
    import copy

    init_state = copy.deepcopy(model.state_dict())
    return


@app.cell
def _(model, nn, torch):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1.2e-3)
    return loss_fn, optimizer


@app.cell
def _(device, loss_fn, model, optimizer, test_loader, torch, train_loader):
    def train(model, loader, loss_fn, optimizer):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def evaluate(model, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
        return correct / len(loader.dataset)


    for epoch in range(5):
        train(model, train_loader, loss_fn, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: test acc = {acc:.4f}")
    return


@app.cell
def _(model, torch):
    masks = {name: torch.ones_like(p) for name, p in model.named_parameters() if "weight" in name}
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
