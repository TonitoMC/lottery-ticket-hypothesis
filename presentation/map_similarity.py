"""¿Se parecen entre sí los mapas de los distintos modelos del ViT (11.6% de los pesos)?

    uv run python presentation/map_similarity.py

Para 150 imágenes de prueba compara, por pares de modelos, el mapa de atención del [CLS] y el mapa de oclusión
(caída de la probabilidad de la clase correcta al tapar cada parche 4x4) con la correlación de Spearman.
Escribe results/map_similarity.json. El código fue generado con ayuda de IA.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "report"))
import make_figures as mf  # noqa: E402

c = mf.vit_ctx()
test, mean, std = c["test"], c["mean"], c["std"]
models = {"denso": mf.vit(c["base"]["dense"]),
          "ticket": mf.vit(c["load"](c["at"]("ticket")[3])["model"]),
          "rewinding": mf.vit(c["load"](c["at"]("late")[3])["model"]),
          "aleatorio": mf.vit(c["load"](c["at"]("random")[3])["model"])}
NC = 150
X = torch.stack([mf.img_tensor(test, i) for i in range(NC)])
Y = torch.tensor(test.targets[:NC])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def norm(x):
    return (x - mean) / std


def occlusion(m, x, y, block=4):
    g = 32 // block
    with torch.no_grad():
        base = torch.softmax(m(norm(x)[None]), 1)[0, y].item()
        xs = []
        for r in range(g):
            for q in range(g):
                z = x.clone()
                z[:, r * block:(r + 1) * block, q * block:(q + 1) * block] = x.mean((1, 2), keepdim=True)
                xs.append(norm(z))
        p = torch.softmax(m(torch.stack(xs)), 1)[:, y]
    return (base - p).reshape(g, g).numpy()


def cls_attention(m, x):
    with torch.no_grad():
        return m.all_attention(norm(x)[None])[-1][0, :, 0, 1:].mean(0).reshape(8, 8).numpy()


with torch.no_grad():
    correct = {n: (m(norm(X)).argmax(1) == Y).numpy() for n, m in models.items()}
all_ok = np.all([correct[n] for n in models], axis=0)
maps = {n: {"att": [], "occ": []} for n in models}
for i in range(NC):
    for n, m in models.items():
        maps[n]["att"].append(cls_attention(m, X[i]).flatten())
        maps[n]["occ"].append(occlusion(m, X[i], int(Y[i])).flatten())

out = {"n_imagenes": NC, "n_todos_aciertan": int(all_ok.sum()), "pares": {}}
for a, b in combinations(models, 2):
    row = {}
    for kind in ("att", "occ"):
        rs = [spearman(maps[a][kind][i], maps[b][kind][i]) for i in range(NC) if maps[a][kind][i].std() > 1e-9 and maps[b][kind][i].std() > 1e-9]
        rs_ok = [spearman(maps[a][kind][i], maps[b][kind][i]) for i in range(NC) if all_ok[i] and maps[a][kind][i].std() > 1e-9 and maps[b][kind][i].std() > 1e-9]
        row[kind] = {"todas": float(np.mean(rs)), "todos_aciertan": float(np.mean(rs_ok)) if rs_ok else None}
    out["pares"][f"{a} vs {b}"] = row
    print(f"{a:10s} vs {b:10s}  atención {row['att']['todas']:+.2f} ({row['att']['todos_aciertan']:+.2f})   oclusión {row['occ']['todas']:+.2f} ({row['occ']['todos_aciertan']:+.2f})")
print("imágenes donde los 4 aciertan:", out["n_todos_aciertan"], "de", NC)
(root / "results" / "map_similarity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
