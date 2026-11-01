# Notes

Working notes from the first full run of notebooks 01-05 (2026-10-30). Not a report, just what we saw, what was decided, and what is still open. Sources are in `REFERENCES.md`.

## First full run (single seed, CUDA, full settings)

Exports: `notebooks/0{1..5}_*.html`. Curves: `results/*.json`.

| Notebook | Dense | Result |
|---|---|---|
| 01 MLP / MNIST | 97.7% | Ticket stays at 97.5-98.2% down to 3.5% of weights. Random reinit falls to 96.0%. |
| 02 CNN / CIFAR-10 | 73.6% | Ticket peaks at 75.9% (64% of weights) and beats random from 80% down to ~33%. At 10.7% the ticket collapses to 49.7%, below random (61.0%). That point is suspect (noise or per-layer pruning of a small layer). |
| 03 ResNet / CIFAR-10 | 90.9% | Plain ticket ~= random (86.3% vs 86.5% at 10.7%), as in [1] at lr 0.1. Rewind to epoch 1 holds 89.6% at 10.7%. |
| 04 ViT / CIFAR-10 | 78.9% | Ticket 82-83% from 42% down to 7.5% of weights (above dense). Random reinit falls to 65.9% at 3.2%. Rewind to epoch 2 ~= plain ticket, slightly lower. |

All curves are one run each. Gaps of about 1 point or less should not be read as real.

## Things to keep straight

- **Augmentation:** 03 and 04 use flip + 4-pixel pad/crop (03 follows the paper, Appendix I.3 of [1]). 01 and 02 use none, as in the paper's small nets.
- **ViT baseline:** 78.9% with augmentation already on, at 20 epochs. It is probably undertrained, not overfit, so "pruning acts as a regularizer" is a hypothesis that has not been tested. The cheap check is dense train vs test accuracy.
- **ViT vs [5]:** [5] reports weight-level tickets are hard to find in ViTs. Our ViT ticket staying above dense is surprising, so hedge it in the report.
- **"Epoch 1 reinit" in 03:** this is a rewind, not a reinit. `rewind_state` is the dense net's weights after 1 full epoch (391 steps). 04 rewinds after 2 epochs.
- **Why 03 is slow:** 32 trainings (1 dense + 10 tickets + 11 random + 10 rewind) of 77 epochs each, ~1h45m. 04 is 26 trainings of 20 epochs, ~45 min.
- **Seeds:** none are set anywhere. CUDA is also not bit-deterministic, so seeds give a shared starting point, not exact reruns.
- **Parameter-count prints:** deliberately removed from the notebooks in an earlier session. `notebooks/03_resnet18_cifar10.py` still has a count cell (`conv_weights` / `total`) and the markdown above it ("Verificamos contra la Figura 2..."). Remove both and re-export 03.

## Open items

1. Remove the parameter-count cells from 03 and re-export (check whether the session cache makes that fast).
2. Multi-seed runs (3+) for 01 and 02; report mean and spread. 03 and 04 stay single-run and are labelled as such.
3. Significance checks: a Wilcoxon signed-rank test over the curve points (ticket vs random) from the existing JSONs; binomial CIs on single accuracies (test-set sampling only); a paired test across seeds once they exist.
4. Re-check 02's 10.7% point.
5. ViT baseline: try 50-100 epochs on the dense model only first, then decide on a full IMP run. SVHN is an optional extra notebook (04b), not a fix for the baseline.
6. Mask analysis in a separate notebook (06), so 01-04 stay free of masks. Needs `lth.py` to return or save the masks (and the trained weights for per-class accuracy).
   - MLP first-layer mask as 28x28 images. The border result is trivial (always-zero pixels never train, so their weights stay small and are pruned first); present it as a sanity check.
   - Remaining weights per layer for 03 and 04 (global pruning).
   - Per-class accuracy, dense vs sparsest ticket.
   - Optional: ViT attention maps; mask overlap across seeds; a random-mask control (random structure, original init) to test whether structure matters beyond the number of weights.
7. Delete the stray empty `notebooks/02_mlp_mnist.py` (if you agree).
