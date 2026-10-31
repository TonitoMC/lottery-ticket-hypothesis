"""Código compartido de LTH (Lottery Ticket Hypothesis) para los notebooks 01-04.

Los notebooks solo llaman a run_lth(...) después de entrenar su modelo denso. El resto de las
funciones las usa run_lth por dentro:

    run_lth
     ├─ run_imp             rondas de IMP (ticket clásico y, si se pide, ticket con rewinding)
     │   ├─ make_masks      máscaras iniciales (todo 1's)
     │   ├─ prune           poda los pesos de menor magnitud
     │   ├─ train_model     re-entrena la subred con sus máscaras
     │   └─ sparsity_report fracción de pesos que sigue viva
     └─ run_random_control  mismas máscaras con pesos aleatorios nuevos (control)

apply_masks pone en 0 los pesos podados; se llama tras cada paso del optimizador en train_model.
"""

import copy

import torch
from torch import nn


# Regresa un tensor de 1's con la forma de cada peso del modelo
# skip: prefijos de capas que no se podan, por ejemplo ("fc",)
def make_masks(model, skip=()):
    return {
        name: torch.ones_like(p)
        for name, p in model.named_parameters()
        if "weight" in name and not name.startswith(tuple(skip))
    }


# Aplica las máscaras, mantiene el valor si la máscara = 1, si no lo regresa a 0
def apply_masks(model, masks):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                # mul_ modifica el parámetro in-place
                p.mul_(masks[name])


# Crea nuevas máscaras que podan el 'rate' % de pesos con menor magnitud
# scope="layer": un corte por capa, scope="global": un solo corte para toda la red
def prune(model, masks, rate, scope="layer"):
    params = dict(model.named_parameters())

    if scope == "global":
        # Magnitudes de todos los sobrevivientes juntos
        survivors = torch.cat([params[n][masks[n] == 1].abs() for n in masks])
        cutoff = survivors.kthvalue(int(rate * survivors.numel())).values
        return {n: masks[n] * (params[n].abs() > cutoff).float() for n in masks}

    new_masks = {}
    for name, p in model.named_parameters():
        if name not in masks:
            continue

        mask = masks[name]

        # Tensor 1D con las magnitudes de los pesos sobrevivientes
        survivors = p[mask == 1].abs()

        k = int(rate * survivors.numel())

        if k == 0:
            new_masks[name] = mask.clone()
            continue

        cutoff = survivors.kthvalue(k).values

        new_masks[name] = mask * (p.abs() > cutoff).float()

    return new_masks


# Fracción de pesos que sigue viva
def sparsity_report(masks):
    total = sum(m.numel() for m in masks.values())
    kept = sum(m.sum().item() for m in masks.values())
    return kept / total


# Entrena un modelo podado desde cero (optimizador nuevo), regresa la accuracy final en prueba
# make_opt: función que crea el optimizador (por defecto Adam con lr)
# lr_at: función época -> lr (por defecto lr constante), start_epoch: época del schedule donde empieza
def train_model(model, masks, train_loader, test_loader, device, epochs, lr, make_opt=None, lr_at=None, start_epoch=0):
    loss_fn = nn.CrossEntropyLoss()
    opt = make_opt(model) if make_opt else torch.optim.Adam(model.parameters(), lr)

    for e in range(epochs):
        if lr_at:
            for g in opt.param_groups:
                g["lr"] = lr_at(start_epoch + e)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Los pesos podados regresan a 0
            apply_masks(model, masks)

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(test_loader.dataset)


# Iterative Magnitude Pruning: podar, regresar a los pesos iniciales, re-entrenar y repetir
# init_state: pesos a los que se regresa tras podar (inicialización o una época temprana)
# **train_args: se pasan a train_model (make_opt, lr_at, start_epoch)
def run_imp(trained_model, init_state, acc, train_loader, test_loader, device, rounds, epochs, lr, rate=0.2, scope="layer", skip=(), **train_args):
    masks = make_masks(trained_model, skip)
    results = [{"round": 0, "weights_left": 1.0, "acc": acc}]
    all_masks = [masks]

    for r in range(1, rounds + 1):
        # Cortar el 20% de menor magnitud de los sobrevivientes
        masks = prune(trained_model, masks, rate, scope)
        # Regresar a los pesos originales
        trained_model.load_state_dict(init_state)
        apply_masks(trained_model, masks)
        acc = train_model(trained_model, masks, train_loader, test_loader, device, epochs, lr, **train_args)

        left = sparsity_report(masks)
        results.append({"round": r, "weights_left": left, "acc": acc})
        all_masks.append(masks)
        print(f"Round {r}: {left:.1%} left, acc = {acc:.4f}")

    return results, all_masks


# Experimento completo de LTH: curva del winning ticket + curva de control con reinicialización aleatoria
# Si se da rewind_state (pesos de la época rewind_epoch) también corre el ticket con rewinding y regresa
# (ticket, late, random); si no, regresa (ticket, random)
#
# Cómo se llama en los notebooks (después de entrenar el modelo denso):
#   01, 02: lth.run_lth(model, init_state, MLP, acc, train_loader, test_loader, device, rounds=15, epochs=5, lr=1.2e-3)
#   03, 04: lo mismo + scope="global", rewind_state=..., rewind_epoch=..., make_opt=..., lr_at=...
#
# model        modelo denso ya entrenado (no se modifica, se trabaja sobre una copia)
# init_state   pesos iniciales guardados antes de entrenar (a los que regresa el ticket clásico)
# make_model   clase del modelo (MLP, CNN, ...), para crear los modelos aleatorios del control
# acc          accuracy del modelo denso (punto de partida de las curvas)
# rounds       rondas de poda; epochs: épocas de re-entrenamiento por ronda; lr: learning rate
# rate         fracción de pesos que sobreviven que se poda en cada ronda (0.2 = 20%)
# scope        "layer" (un corte por capa) o "global" (un corte para toda la red)
# skip         prefijos de capas que no se podan, por ejemplo ("fc",)
# rewind_state, rewind_epoch   pesos guardados tras rewind_epoch épocas del entrenamiento denso
# make_opt, lr_at              optimizador y schedule de lr (por defecto Adam con lr constante)
def run_lth(model, init_state, make_model, acc, train_loader, test_loader, device, rounds, epochs, lr,
            rate=0.2, scope="layer", skip=(), rewind_state=None, rewind_epoch=0, **train_args):
    print("Winning tickets:")
    ticket_results, all_masks = run_imp(copy.deepcopy(model), init_state, acc, train_loader, test_loader, device,
                                        rounds, epochs, lr, rate, scope, skip, **train_args)
    print("Random reinit:")
    random_results = run_random_control(make_model, all_masks, train_loader, test_loader, device, epochs, lr, **train_args)

    if rewind_state is None:
        return ticket_results, random_results

    print(f"Winning tickets (rewind a la época {rewind_epoch}):")
    late_results, _ = run_imp(copy.deepcopy(model), rewind_state, acc, train_loader, test_loader, device,
                              rounds, epochs - rewind_epoch, lr, rate, scope, skip, start_epoch=rewind_epoch,
                              **{k: v for k, v in train_args.items() if k != "start_epoch"})
    return ticket_results, late_results, random_results


# Mismas máscaras que los tickets, pero con una inicialización aleatoria nueva
def run_random_control(make_model, all_masks, train_loader, test_loader, device, epochs, lr, **train_args):
    results = []
    for r, masks in enumerate(all_masks):
        m = make_model().to(device)
        apply_masks(m, masks)
        acc = train_model(m, masks, train_loader, test_loader, device, epochs, lr, **train_args)

        left = sparsity_report(masks)
        results.append({"round": r, "weights_left": left, "acc": acc})
        print(f"Round {r}: {left:.1%} left, acc = {acc:.4f}")

    return results
