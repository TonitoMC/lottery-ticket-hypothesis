"""Genera slides.html: mete los resultados de results/*.json (y, si falta, del log de la corrida del ViT largo) en template.html.

    python3 presentation/build.py
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def load(name):
    p = root / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pairs(res):
    return [[round(r["weights_left"] * 100, 2), round(r["acc"], 4)] for r in res]


def from_json(d):
    if not d:
        return None
    return {k: pairs(v) for k, v in d.items()}


# Mientras el ViT largo no termina no hay JSON: se leen las rondas del log
def from_log(path):
    if not path.exists():
        return None
    curves, cur = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("Winning tickets (rewind"):
            cur = "ticket_late"
        elif line.startswith("Winning tickets"):
            cur = "ticket"
        elif line.startswith("Random reinit"):
            cur = "random"
        m = re.match(r"Round (\d+): ([\d.]+)% left, acc = ([\d.]+)", line)
        if m and cur:
            curves.setdefault(cur, []).append([float(m[2]), float(m[3])])
    m = re.search(r"Dense final: test acc = ([\d.]+)", path.read_text())
    if m and "ticket" in curves:
        curves["ticket"].insert(0, [100.0, float(m[1])])
    return curves or None


vit_long = from_json(load("vit_longer"))
data = {
    "mlp": from_json(load("mlp_mnist")),
    "cnn": from_json(load("cnn_cifar10")),
    "cnng": from_json(load("cnn_global_cifar10")),
    "resnet": from_json(load("resnet18_cifar10")),
    "vit": from_json(load("vit_cifar10")),
    "vitlong": vit_long,
    "vitlong_partial": not (root / "results" / "vit_longer.json").exists(),
}
html = (Path(__file__).parent / "template.html").read_text().replace("/*DATA*/null", json.dumps(data))
(Path(__file__).parent / "slides.html").write_text(html)
print("slides.html escrito; ViT largo:", "parcial" if data["vitlong_partial"] else "completo", "" if vit_long else "(sin datos)")
