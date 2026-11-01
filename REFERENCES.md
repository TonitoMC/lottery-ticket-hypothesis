# References

Sources the experiments are based on. Each entry says what we took from it and where.
"Checked" = the relevant section/table was read; "Consulted" = only the abstract/summary was seen.

## [1] Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019
https://arxiv.org/abs/1803.03635 (checked: Fig. 2, Sec. 4, Appendix)

- Defines IMP with reset to initialization (`lth.run_imp`, `rewind_epoch=0`) and the random-reinit control.
- **LeNet-300-100 / MNIST** (nb 01): Adam `1.2e-3`, batch 60, 50K iterations, prune 20% per round (10% for the output layer).
- **Conv-2/4/6 / CIFAR-10** (nb 02): Conv-4 uses Adam `3e-4`, batch 60, 25K iterations, per-layer pruning of conv 10% / fc 20%.
- **ResNet-18** (nb 03; Appendix I.3, Fig. 2, Sec. 4): their "Resnet-18" is the 20-layer CIFAR ResNet of He et al. 2016 (conv + nine residual pairs, 16/32/64 channels, 274K params), not the ImageNet ResNet-18. Recipe: SGD momentum 0.9, batch 128, weight decay `1e-4`, flip + 4-pixel pad/crop augmentation, Glorot-normal init, `lr = 0.1` divided by 10 at 20K and 25K of 30K iterations, global pruning of conv layers only at 20% per round (fc not pruned). At `lr = 0.1` no winning tickets were found; they appeared at `lr = 0.01`, or `lr = 0.03` with 20K iterations of warmup. Our 1x1-conv shortcuts reproduce their 270K conv weights / 274K total exactly.

## [2] Frankle, Dziugaite, Roy & Carbin, "Stabilizing the Lottery Ticket Hypothesis" (v1 title: "The Lottery Ticket Hypothesis at Scale"), 2019
https://arxiv.org/abs/1903.01611 (checked: Sec. 3-4; later extended as "Linear Mode Connectivity and the Lottery Ticket Hypothesis", https://arxiv.org/abs/1912.05671, consulted)

- Introduces IMP with rewinding to iteration `k` instead of 0 (`rewind_epoch` in nb 03/04).
- At the standard `lr = 0.1`, resetting to iteration 0 does not give winning tickets for ResNet-18 and VGG-19, but rewinding to iteration 500 (ResNet-18) or 100 (VGG-19) gives subnetworks that exceed the original accuracy (Sec. 4). Nb 03 reproduces this setting with `REWIND_EPOCH = 1` (391 iterations, the closest epoch to 500).
- Retraining after a rewind to `k` runs for the remaining `T - k` iterations; our `fit(start_epoch=k)` does the same.

## [3] Renda, Frankle & Carbin, "Comparing Rewinding and Fine-tuning in Neural Network Pruning", ICLR 2020
https://arxiv.org/abs/2003.02389 (consulted)

- Reference CIFAR-10 ResNet-56 recipe: Nesterov SGD, momentum 0.9, batch 128, weight decay `2e-4`, 182 epochs. Used only as context for SGD + weight decay on CIFAR ResNets (our `5e-4` is not taken from here).

## [4] Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT), ICML 2021
https://arxiv.org/abs/2012.12877 (checked: Table 9)

- ViT recipe for training from scratch: AdamW, weight decay `0.05`, 5 warmup epochs, `lr = 5e-4 x batch/512`. Nb 04 uses AdamW, weight decay `0.05` and warmup + cosine from this.
- **Deviation:** the rule above gives `1.25e-4` at batch 128; we use `1e-3` (not from this source, untuned).

## [5] Shen, Kong, Qin, Dong, Yuan, Meng, Tang, Ma & Wang, "Data Level Lottery Ticket Hypothesis for Vision Transformers", IJCAI 2023 (arXiv v1: 2022)
https://arxiv.org/abs/2211.01484 (checked: abstract and introduction)

- Reports that conventional weight-level winning tickets are hard to find in DeiT-Small (ImageNet) even with rewinding. Nb 04 may therefore show no ticket at all; that would be consistent with this source, not necessarily a bug.

## [6] Chen et al., "Chasing Sparsity in Vision Transformers: An End-to-End Exploration" (SViTE), NeurIPS 2021
https://arxiv.org/abs/2106.04533 (consulted)

- Sparse training of DeiT (not post-hoc IMP). Background only; nothing is taken from it.

## Where our setup departs from the sources

| Item | Source | Ours |
|---|---|---|
| Epochs per round, nb 01 | 50K iterations [1] | 5 epochs (~5K iterations) |
| Output-layer rate, nb 01 | 10% [1] | 20% |
| Conv-4 prune rates, nb 02 | conv 10% / fc 20% [1] | 20% everywhere |
| Conv-4 batch, nb 02 | 60 [1] | 64 (30 epochs, ~25K iterations as in [1]) |
| ResNet, nb 03 | 20-layer CIFAR ResNet [1] | same network, built by hand (270,256 conv weights, 274,042 params + BN buffers) |
| ResNet recipe, nb 03 | Appendix I.3 of [1] | same: 77 epochs for ~30K iterations, lr drops at epochs 51 and 64 |
| Validation split, nb 03 | 45K train / 5K val [1] | all 50K for training, no validation set |
| Rewind point, nb 03 | iteration 500 [2] | epoch 1 = iteration 391 (rewinding is per epoch) |
| Normalization, nb 03 | per-pixel mean subtraction | per-channel mean / std |
| Optimizer state at rewind, nb 03/04 | continuous | momentum / Adam state restarts at the rewind epoch |
| Prune rate per round, nb 04 | 20% [1] | 35% (fewer rounds to reach ~3%) |
| ViT `lr`, nb 04 | `5e-4 x batch/512` = `1.25e-4` [4] | `1e-3` (untuned) |
| ViT warmup, nb 04 | 5 epochs [4] | 2 epochs of 20 |
| Seeds | averaged over runs | one run |
