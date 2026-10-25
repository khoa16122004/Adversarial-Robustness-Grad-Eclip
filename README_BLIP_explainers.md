# BLIP explainers and adversarial causal evaluation

`test_blip_explainers.py` lives at the repository root and uses the existing
`BLIP` source folder relative to the script, independently of the working
directory. It does not import attack code, CLIP models, or the global
model-loading code in `generate_emap.py`.

Code stays in this repository. The E: drive is only for data:

- Default checkpoint:
  `E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model\BLIP.pth`
- Default output directory:
  `E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model\explainer_results`

Override these with `--checkpoint` and `--output` if needed. No scripts need to
be copied to E:.

## Environment

Use a separate Python 3.10 environment with a compatible PyTorch/torchvision pair,
NumPy, Pillow, `timm==0.4.12`, `transformers==4.26.1`, and `fairscale==0.4.13`.
These are BLIP-era dependencies; recent transformers/timm releases can remove
APIs used by the original BLIP source. PyTorch must support
`torch.load(..., weights_only=True)` (PyTorch 2.x recommended).
No packages or weights are installed/downloaded by this runner explicitly.
BLIP's tokenizer initialization may download `bert-base-uncased` on first use;
pre-cache it when running offline. Do not use untrusted checkpoints.

`BLIP/models/med.py` imports the chunking/pruning helpers from
`transformers.pytorch_utils`, with a legacy fallback for Transformers 4.15.
If the server reports that `apply_chunking_to_forward` cannot be imported from
`transformers.modeling_utils`, update this BLIP source file along with the runner.
This import fix does not certify compatibility with every newer Transformers
release; full model execution still needs validation in the server environment.

If `find_pruneable_heads_and_indices` is also missing from
`transformers.pytorch_utils`, the import-path patch alone is insufficient.
Install the pinned Transformers compatibility dependency in the environment
used by the launcher, then resubmit:

```bash
conda activate Sparse_Attack
python -m pip install -r requirements_blip_compat.txt
sbatch script_sh/test_blip.sh
```

The launcher checks the pinned version and all three helpers before model
loading; it never installs packages automatically. Changing Transformers can
affect other projects sharing `Sparse_Attack`; use a separate BLIP environment
and update the launcher's `conda activate` line if those projects require a
different version.

The local checkpoint must have BLIP ITC weights (`visual_encoder`, `text_encoder`,
`vision_proj`, `text_proj`). Momentum/queue and other unused checkpoint keys are
reported but permitted. Missing ITC weights or incompatible shapes fail loudly.
Use `--vit large` if the checkpoint is ViT-L rather than ViT-B.

## Run (PowerShell)

```powershell
Set-Location D:\Adversarial-Robustness-Grad-Eclip
python .\test_blip_explainers.py `
  --blip-root D:\Adversarial-Robustness-Grad-Eclip\BLIP `
  --image D:\Adversarial-Robustness-Grad-Eclip\images\ostrich.png `
  --text "an ostrich" `
  --output E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model\results_ostrich
```

The default runs GradECLIP, GAME, GradCAM, MaskCLIP, and RISE.
For a quick smoke run, add `--rise-masks 32 --batch-size 4`.
The default RISE run uses 2,000 masks; 32 masks are only for checking execution,
not for assessing explanation quality. Reduce `--batch-size` if GPU memory is
limited. Use `--device cpu` to force CPU.

RISE defaults match the CLIP runner's numerical settings: 2,000 masks, 8x8
coarse grid, keep probability 0.5 and inference batch size 50. The BLIP mask
implementation still uses seeded PyTorch sampling/bilinear resizing rather than
CLIP's NumPy/skimage implementation, so the sampled masks are not identical.

Select individual explainers with, for example:

```powershell
python .\test_blip_explainers.py --image "D:\path\to\example.jpg" `
  --text "a dog on the grass" --methods gradeclip game gradcam maskclip `
  --output E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model\results_dog
```

## Adaptation and score target

### Linux / Slurm launcher

Run `script_sh/test_blip.sh` from the server repository. It explicitly passes
the server BLIP source, checkpoint, and output paths, without using Windows
defaults. The default checkpoint is `blip_checkpoint.pth` at the server repo
root; change it via `CHECKPOINT`. Results default to
`BLIP_faithfulness/results_<job-id>` inside that repo.
Create the Slurm log directory before submission:

```bash
mkdir -p BLIP_faithfulness
CHECKPOINT="/path/to/BLIP.pth" OUTPUT_DIR="/path/to/results" sbatch script_sh/test_blip.sh
```

The launcher retains the 32-mask RISE smoke test; increase it for quality checks.

### Methods

- All methods explain the same **image-text contrastive cosine (ITC)** for one
  caption, not the multimodal ITM classification head.
- Default `--last-heads 1` follows `gradeclip_adaptation_blip.ipynb`: preceding
  blocks retain native attention, the final block uses a single concatenated
  head, and the pretrained attention scale is retained. This **changes BLIP's
  forward score**. It is not presented as the untouched pretrained model.
- Use `--last-heads native` to keep pretrained multi-head attention in every
  block. The runner checks its custom image forward against the original BLIP
  visual encoder before producing explanations.
- GradECLIP uses the gradient at the final pre-output-projection attention
  result, value vectors, and min-max normalized CLS-query/patch-key cosine
  locality, matching the notebook in single-head mode.
- GAME adapts the repository's positive gradient-weighted attention relevance
  update `R = R + cam @ R`. Default `--game-start-layer -1` uses the final
  block, as in the repository; `--game-start-layer 0` rolls out all blocks.
  This is an ITC vision-attention adaptation, not cross-attention ITM GAME.
- GradCAM uses the input to the final visual block, matching the repository's
  pre-final-block CAM and token-averaged gradient weights (including CLS).
- MaskCLIP is a BLIP adaptation of the repository's value-only final-block
  reconstruction, projected through BLIP's visual normalization/projection,
  then weighted by projected CLS-key/patch-key cosine. It is not an official
  MaskCLIP BLIP implementation or simply final patch-token similarity.
- RISE uses seeded random coarse masks, bilinear upsampling and random crops,
  weighted by the same ITC cosine. As in the repository, masks multiply the
  normalized image: masked-out regions correspond to the RGB normalization
  mean, not black. Raw maps may be signed because ITC cosine can be negative.

The input is resized to a square (`--image-size 224`, divisible by 16), as in the
notebook's fixed-resolution preprocessing. All overlays are mapped back to the
original image dimensions. This runner does not implement keep-aspect-ratio or
variable-resolution notebook preprocessing.

## Outputs and validation

Each output folder contains:

- `input.png`, `comparison.png`;
- `<method>_raw.npy`: unnormalized patch-grid map (full model resolution for RISE);
- `<method>_heatmap.npy`: finite, normalized `[0, 1]` map at original image size;
- `<method>_heatmap.png`, `<method>_overlay.png`;
- `summary.json`: input/caption/checkpoint, settings, native and explained ITC
  scores, elapsed explainer time, map shapes/ranges and constant-map indicators.

Constant maps are reported with a warning and normalized to zero. Nonfinite
outputs fail. Errors propagate with a traceback; the runner does not replace a
failed method with a dummy success map. Choose a separate output folder for each
image/caption/configuration to avoid overwriting results or mixing old outputs.

No inference or numerical validation was performed when preparing these files:
the user will run them in their BLIP environment. In particular, the local
checkpoint architecture and the proposed dependency combination still need
runtime verification.

## Adversarial evaluation: `adv_score_BLIP.py`

This is a separate BLIP entry point; `adv_score_CLIP.py` and
`RISE/evaluation.py` remain unchanged. It reuses the existing
`AdversarialCausalMetric`, `JointAdversarialCausalMetric`, and `CausalMetric`
engines, including the original PGD loss, update directions, best-iterate
selection and post-attack deletion/insertion reruns.

`blip_attack_adapter.py` supplies BLIP image/text encoding, zero-shot classifier
weights and an explainer callback with the same interface as the CLIP runner.
It reuses the model loader and explanation implementations from
`test_blip_explainers.py`. The only change in shared `util.py` is deferring
CLIP-specific imports until the CLIP functions need them (and removing an unused
CLIP import). Existing utility signatures and calculations are unchanged.

### Run on the server

Activate the BLIP-compatible environment described above, then run from the
repository root:

```bash
python adv_score_BLIP.py \
  --blip-checkpoint /path/to/BLIP.pth \
  --img-dir /path/to/ImageNet/val \
  --sample-path /path/to/samples.json \
  --output-dir /path/to/BLIP_model/attack_results \
  --hm-type eclip \
  --mode del \
  --eps 32 --alpha 8 --pgd-steps 50 \
  --process-batch-size 32
```

The sample JSON has the same format as CLIP:

```json
{
  "sample_001": "n01518878/example.JPEG",
  "sample_002": "n02099601/example.JPEG"
}
```

Each value is a path relative to `--img-dir`. Class names default to the same
ImageNet class order used by CLIP; use `--classnames-path` for custom classes,
with the same TXT/JSON formats accepted by `util.load_classnames`.

- `--hm-type`: `eclip` (alias `gradeclip`), `game`, `gradcam`, `maskclip`, `rise`.
  CLIP-only explainers are deliberately not accepted.
- `--mode`: `del`, `ins`, or `del+ins`.
- `--last-heads 1` is the notebook adaptation; `--last-heads native` keeps native
  attention. **Both classifier and explainer use this same setting**, unlike a
  configuration mixing native classifier scores with modified explanations.
- Classifier logits are `100 * normalized_image_features @ text_weights`, as in
  the CLIP utility. Classification uses `--prompt-template` (default
  `a photo of {}`), while saliency uses the bare target class name, matching
  `adv_score_CLIP.py`. No trained BLIP temperature is substituted silently.
- `--target-source pred` is the default. `gt` requires `--gt-label`, which is a
  single explicit label for all samples (same as CLIP; not inferred from paths).
  This controls the explanation caption. The unchanged PGD engines optimize
  and preserve the **clean predicted class**, and clean post-attack reruns
  score the **adversarial predicted class**, as in the CLIP pipeline.
  JSON records the explanation target separately from the retained class.
- This runner requires **CUDA and 224x224 images**, because shared causal
  engines use `.cuda()` and `HW = 224 * 224`. `--device cuda:N` selects the
  matching default CUDA device for those shared calls.
- BLIP weights are frozen, but image gradients remain enabled. The explainer
  computes gradients on a detached image copy so it cannot consume the PGD
  graph; no differentiation through saliency ranking is introduced.
- `--rise-masks`, `--rise-grid`, `--rise-prob`, `--rise-batch-size`,
  `--game-start-layer`, `--seed` are available. The process batch size controls
  causal path scoring, **not** RISE inference. RISE recomputes maps every PGD
  iteration and can be expensive; its masks repeat deterministically for a seed.
- `--save-process` saves deletion/insertion rerun previews, as with CLIP.
  Decrease `--process-batch-size` if a forward batch is too large. The shared
  engine still retains the path's computation graphs for the PGD backward pass.

For a smoke run, use a JSON containing one image and add
`--pgd-steps 2 --step 25088 --process-batch-size 2`.
For RISE smoke testing also add `--rise-masks 32 --rise-batch-size 4`.
These coarse settings validate execution, not explanation quality.

### Output parity with CLIP

The directory structure is:

```text
<output-dir>/<hm-type>/<mode>/
  run_config.json
  <sample-id>/
    adversarial_image_del.png          # or _ins.png / _del_ins.png
    del_del_adv_eclip_saliency.npy     # corresponding heatmap and overlay PNGs
    del_ins_adv_eclip_saliency.npy
    del_del_summary.png
    del_ins_summary.png
    curve_information.json
```

For insertion use prefix `ins`, and for joint mode use prefix `joint`, exactly
as in CLIP. Each saliency stem has `.npy`, `.png`, and `_overlay.png` outputs.
With `--save-process`, previews go in `steps_<prefix>_del` and
`steps_<prefix>_ins`.

`curve_information.json` preserves the CLIP keys for clean/adversarial
probabilities, retained class, adversarial prediction, `pgd_loss`, `pgd_trace`,
and deletion/insertion curves and AUC. Joint mode includes `combined_score`
and `score_formula`. Additional fields identify BLIP, the explanation target,
clean prediction, and selected PGD iterate. `run_config.json` records model,
checkpoint and attention/attack/explainer settings. Use different output roots
for different checkpoints or attention settings to avoid overwrites.

The output image is checked for finite values and the requested L-infinity
bound before saving. Tests use a tiny BLIP-shaped model with real PyTorch
autograd (no checkpoint/downloads), cover all five explainers in both attention
modes and the shared PGD engines and artifact writers in all three attack modes:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p test_blip_attack.py -v
```

These contract tests do not replace a full-checkpoint/GPU run on the server.

### Server / Slurm sweep launcher

`script_sh/run_blip.sh` runs all 15 combinations of
`eclip`, `game`, `gradcam`, `maskclip`, `rise` and pixel epsilons `2`, `4`, `8`.
It sets `alpha = epsilon / 4`, joint `del+ins`, 50 PGD iterations and process
batch size 100, matching the supplied CLIP sweep. It follows `test_blip.sh`:
the same Slurm resources, server repository path, `Sparse_Attack` conda
environment and certificate settings. It uses distinct attack log filenames.

Submit from the server repository root (the log directory must exist before
submission):

```bash
mkdir -p BLIP_faithfulness
sbatch script_sh/run_blip.sh
```

The default `IMG_DIR` is `/datastore/quanphm/datasets/ImageNet/val`; override
`IMG_DIR` to evaluate another dataset. The checkpoint defaults to `blip_checkpoint.pth` in the
server repository root, as in `test_blip.sh`; override with `CHECKPOINT`.
The sample JSON is `classification_reuslt/pet/vit_b_16_100_random_50.json`.
This JSON selects images originally selected with CLIP; it does not imply
BLIP classifies all of them correctly. Classifier class names remain ImageNet
by default, just as in the supplied CLIP command (no automatic switch to a
120-class Dogs classifier).

Optional environment overrides: `IMG_DIR`, `SAMPLE_PATH`, `OUTPUT_DIR`,
`PROCESS_BATCH_SIZE`, `RISE_MASKS`, `RISE_BATCH_SIZE`, `LAST_HEADS`.
For example, lower `PROCESS_BATCH_SIZE` if GPU memory is insufficient:

```bash
IMG_DIR="/path/to/stanford-dogs/Images" CHECKPOINT="/path/to/BLIP.pth" \
  PROCESS_BATCH_SIZE=4 sbatch script_sh/run_blip.sh
```

Outputs default to
`BLIP_faithfulness/attack_<job-id>/eps_<epsilon>/<hm>/del+ins/<sample-id>`
under the server repository.
The launcher avoids adding `<hm>` twice since the Python runner already adds it.
After all runs succeed it creates
`BLIP_faithfulness/attack_<job-id>.zip`. It stops on the first error
and does not report a partial sweep as completed. No dependencies are installed
automatically. Check the environment instructions above before launching; the
script checks Transformers helper imports and CUDA availability.
