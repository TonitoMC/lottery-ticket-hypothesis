# Pending

What is left as of 2026-10-31. `NOTES.md` has the first-run results and the original open items; this file is the current to-do list. Not decided yet = marked **(decide)**.

## Due tomorrow: what actually gets done

Deadline is tomorrow, so the list below is cut down. Everything else in this file is "future work" and gets one line in the report.

1. **ViT 05 run** finishes on its own (~22:00). Then open `06_inspection.py` on it and take the plots.
2. **CNN 02 re-run, once** (no seeds). Only to see whether the 10.7% collapse reproduces. Checkpoints for 06 section 1 come for free.
3. **Update `05_comparison.py`** to include `results/vit_longer.json`, and write the report from `NOTES.md` plus the final numbers.
4. **Claims we can make:** rewinding is needed for the ResNet (k=1 beats k=0 by ~3 points at 10.7%), not for the MLP, and for the ViT the plain ticket already works (04, and 05 once finished). "More complex means later" is only supported qualitatively, from one or two rewind points per architecture. Say that plainly in the report, single seed, and don't claim more.
5. **Dropped (future work):** the rewind-epoch sweep, CNN/MLP rewinding, seeds + significance tests, re-running 01/03/04 for checkpoints, the 03 cell cleanup, the numbering rename.

## Running now

- **`05_vit_longer.py`** (100 epochs, RandAugment + random erasing + mixup/cutmix + label smoothing, rewind to epoch 5). Started 2026-10-31 ~18:00, expected to finish around 22:00. Log: `vit_longer.log`. Checkpoints: `checkpoints/05_vit_longer/` (written after every round).
  - Dense baseline: 85.95% test (04 had 78.9%), train 90.3% on clean images.
  - Ticket so far is ~3 points above dense (87.9-89.0% at 65-18% of weights). Wait for the random curve before reading anything into it.
- When it finishes: open `06_inspection.py` on that folder and look at all four sections.

## Experiments still missing

1. **CNN (02) re-run, with seeds.** Check whether the 10.7% collapse (49.7%, below random 61.0%) reproduces. Look at per-layer survivors in `06_inspection.py` section 1 (02 prunes per layer, so a layer that deviates is suspect). Needs a seed loop, since no seeds are set anywhere.
2. **CNN rewinding / epoch resets.** 02 only resets to init. Add a rewind curve so it can be compared with 03 and 04.
3. **MLP rewinding (optional).** The plain ticket already matches dense, so this would just show a flat line.
4. **Rewind-epoch sweep across all four architectures (decide).** Same k values (e.g. 0, 1, 2, 5, 10) at one fixed sparsity, accuracy vs k per architecture. This is what tests the proposal's hypothesis ("the more complex the network, the later the ticket appears"). Today the evidence is ResNet k=0 vs k=1, ViT k=0 vs k=2 (04) and k=5 (05), MLP/CNN nothing. Needs the dense runs to save snapshots at several epochs, not just one.
5. **Significance** (from `NOTES.md`): Wilcoxon over curve points (ticket vs random) from the JSONs, binomial CIs on single accuracies, paired test once seeds exist.

## Re-runs for checkpoints

01, 02, 03 and 04 were run before checkpoint saving existed, so they have no weights or masks to inspect. All four notebooks now call `lth.save_base` and pass `save_dir`, so a re-run saves them. Rough times: 01 minutes, 02 maybe 30-60 min, 03 ~1h45m, 04 ~45 min. Re-runs overwrite `results/*.json` (which `05_comparison.py` reads) and the numbers will shift a little, since there are no seeds. Run them one at a time, not alongside the ViT job.

- 03: remove the parameter-count cell and the markdown above it (`03_resnet18_cifar10.py` ~lines 209-220) before re-exporting.
- Re-export the HTML for any notebook that changes.

## Code

- **`06_inspection.py`**: sections 2-4 (per-class accuracy, attention, gradients) only work for the ViT because the model class lives in the notebook. Sections 1 (mask survival) works on any checkpoint folder. Paste in the CNN / ResNet / MLP classes if those need the other sections.
- Only tested on a toy checkpoint folder so far, not on real data.

## Writing

- **Report notebook** built from `NOTES.md`, with plots from the JSONs.
- **Numbering (decide):** `05_vit_longer.py`, `05_comparison.py` and `06_inspection.py` now exist. Rename `05_comparison.py` (e.g. to `07_comparison.py`) or leave as is.
- Hedge the ViT-above-dense result in the report ([5] says weight-level tickets are hard to find in ViTs).

## Cleanup

- Delete the empty `notebooks/02_mlp_mnist.py`.
- `vit_longer.log` and `vit_longer_dense_first_attempt.log` in the project root are untracked.

## Not doing

- Cross-dataset ticket transfer.
- SVHN (04b).
