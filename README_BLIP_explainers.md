# BLIP explainer-only runner

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

Select individual explainers with, for example:

```powershell
python .\test_blip_explainers.py --image "D:\path\to\example.jpg" `
  --text "a dog on the grass" --methods gradeclip game gradcam maskclip `
  --output E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model\results_dog
```

## Adaptation and score target

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
