import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proyecto Final
    Este Notebook es parte del Proyecto Final del curso de Deep Learning, este proyecto trata sobre un fenómeno conocido como el "Lottery Ticket Hypothesis".

    ## 02 - CIFAR-10 CNN

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

    # Set GPU - CPU as device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return datasets, device, mo, nn, plt, torch, v2


@app.cell
def _(datasets, plt, torch, v2):
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

    _img, _label = cifar_train[0]
    plt.imshow(_img.permute(1,2,0))
    plt.title(_label)
    return cifar_test, cifar_train


@app.cell
def _(cifar_test, cifar_train):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(cifar_train, 64, shuffle=True)
    test_loader = DataLoader(cifar_test, 64, shuffle=False)

    x, y = next(iter(train_loader))

    print(f"Feature batch shape: {x.shape}")
    print(f"Label batch shape: {y.shape}")
    return test_loader, train_loader


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
                # [64, 16, 16]
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
            )
            self.linear = nn.Sequential(
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
    return (cnn,)


@app.cell
def _(device, nn, test_loader, torch):
    def eval(model, loader):
        model.eval()
        correct = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
        return correct / len(loader.dataset)

    def train(model, epochs, lr, loader):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for e in range(epochs):
            for x, y in loader:
                # Move to GPU
                x, y = x.to(device), y.to(device)
                # Generate Logits
                logits = model(x)
                # Calculate Loss
                loss = loss_fn(logits, y)
                # Reset Grad
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # Paso de optimizador
                optimizer.step()

            print(f"Epoch: {e}, Val Loss: {eval(model, test_loader)}")


    return eval, train


@app.cell
def _(cnn, train, train_loader):
    train(cnn, 30, 3e-4, train_loader)
    return


@app.cell
def _(cnn, eval, test_loader):
    eval(cnn, test_loader)
    return


if __name__ == "__main__":
    app.run()
