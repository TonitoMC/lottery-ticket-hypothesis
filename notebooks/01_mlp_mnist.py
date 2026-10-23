import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md("# MLP on MNIST")
    return


if __name__ == "__main__":
    app.run()
