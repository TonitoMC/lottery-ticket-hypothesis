# Pending

As of 2026-10-31 (night). All training is finished except one optional run. Everything else left is writing.

## 1. Run the pending experiment (optional): `02b_cnn_global.py`

Repeats the CNN with **global** instead of per-layer pruning, to check whether that is why the 10.7% point collapses (ticket 53.0% vs. random 61.2%, reproduced in a second run; masks at that point show 7/64 empty filters in the first conv and 102/256 empty hidden units).

- About 75 minutes alone on the GPU. Writes `results/cnn_global_cifar10.json` and `checkpoints/02b_cnn_global/`. Its last cell plots per-layer (02) vs. global.
- If it is **not** run: the report already words the cause as "possible, not proven". Nothing else breaks.
- If it **is** run: update the CNN paragraph in `07_informe.py` (section 4), the CNN text on slide 6 (`presentation/template.html`, `t-cnn`) and discussion question 7, then rebuild (commands at the bottom).

## 2. Your analysis (the part that is yours)

In `notebooks/07_informe.py`, every **[POR COMPLETAR]** block, with the guiding questions already written under it:

- Section 5, after the long-ViT table: why rewinding only helps at the sparsest points, and how that relates to 04.
- Section 6, after the figures: layer survival (`head` / `patch_embed` keep more), the random control's attention concentrating at the bottom, gradient norms at step 0, per-class differences.
- Section 7 (Discusión): read the "Qué sí apoyan" and "Limitaciones" bullets and edit to your own words.
- Section 8: the 12 discussion questions. Keep, cut or reword; each says which experiment would answer it.
- `06_inspection.html` has more views than the report shows (qkv per head, ticket-vs-rewind IoU, mask shapes, entropy). Pick what you want to discuss.

Then transcribe to LaTeX. The figures to reuse are in `presentation/img/*.png`; the numbers come from `results/*.json`.

Slides (`presentation/template.html`):

- Slide 5 still has a "Agregar los prompts de sesiones anteriores" chip: paste your prompts there and add any detail on marimo / Claude Code.
- Slide 9 only describes the figures; add your reading if you want it on the slide.
- Re-read slide 10 against your conclusions.

## 3. Things to keep straight when writing

- **Augmentation:** 03, 04 use flip + 4-pixel pad/crop (03 follows [1], Appendix I.3). 01, 02 use none. 05 adds RandAugment, random erasing, mixup/cutmix and label smoothing.
- **Epoch 1 in 03 is a rewind, not a reinit:** the dense net's weights after 1 epoch (391 steps). 04 rewinds after 2 epochs, 05 after 5 (end of warmup).
- **ViT vs [5]:** [5] says weight-level tickets are hard to find in ViTs. Ours stay above dense, so hedge it (pruning-as-regularizer is untested).
- **One seed everywhere, CUDA not deterministic.** Differences of ~1 point or less are noise. The ticket-vs-random gaps (5-9 points) are not.
- **Pruning scope:** 01, 02 per layer; 03, 04, 05 global. The LayerNorm weights in the ViT are pruned too.
- **Checkpoints exist only for** 01, 02 and 05; 03 and 04 predate the saving code (their inspection would need a rerun, ~1h45m and ~45m).
- **Cost:** ~10 GPU-hours in total, ~4 kWh.

## 4. Not done, to mention as future work

Rewind-epoch sweep (the actual test of "more complex means later"), several seeds + significance tests (McNemar on saved predictions is the cheap one), warmup on/off for ResNet and ViT, cross-dataset transfer, SVHN, a permuted-values control for the random reinit.

## 5. Loose ends

- Empty `notebooks/02_mlp_mnist.py` can be deleted.
- `05_comparison.py` is numbered like `05_vit_longer.py`; rename (e.g. to `08_comparison.py`) or leave.
- The `*.log` files in the project root are tracked or untracked at random; add `*.log` to `.gitignore` if you do not want them in git.
- `03_resnet18_cifar10.py` still has a parameter-count cell (~lines 209-220), harmless.

## Commands

```bash
python3 presentation/build.py                                   # slides from results/*.json
cd notebooks && uv run python ../presentation/make_figs.py      # slide 9 figures from checkpoints
cd notebooks && uv run marimo export html 07_informe.py -o 07_informe.html
cd notebooks && INSPECT_DIR=../checkpoints/05_vit_longer INSPECT_PCT=10 uv run marimo export html 06_inspection.py -o 06_inspection.html
```
