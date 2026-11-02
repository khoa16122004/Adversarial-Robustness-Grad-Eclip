# SigLIP explainers and adversarial evaluation

The SigLIP implementation uses Hugging Face `SiglipModel` and
`SiglipProcessor`; no separate checkpoint file or BLIP source checkout is
needed. Use a Transformers installation that includes SigLIP (see
`requirements_siglip.txt`). Keep this dependency in a separate environment
from the BLIP environment pinned to Transformers 4.26.1.

The default model is `google/siglip-base-patch16-224`. Other Hugging Face model
IDs or local model directories can be supplied with `--model`, but the shared
causal attack engine currently requires square 224x224 model inputs.

## Test explainers on one image

```bash
python test_siglip_explainers.py \
  --model google/siglip-base-patch16-224 \
  --image /path/to/image.jpg \
  --text "a photo of a dog" \
  --output /path/to/siglip_explainer_results \
  --methods gradeclip game gradcam maskclip rise \
  --device cuda \
  --rise-masks 2000 \
  --batch-size 50
```

The script checks its custom image encoder against `SiglipModel.get_image_features`
before generating the maps. It saves each method's NumPy map, heatmap, and
overlay, plus `summary.json`. Saved maps and overlays are resized to the
original image dimensions; the summary records both patch-map and output shapes.
SigLIP has no class token in its vision patch sequence. The learned pooling
probe is its image-level query; the explainers use that probe instead of
inventing an average patch query.
The image encoder uses SigLIP's post-layer normalization and learned attention
pooling head, not CLIP's pre-layer normalization or visual projection.

The `siglip-pooling-v3-headwise` adaptations are:

- Grad-ECLIP uses the gradient of cosine similarity with respect to the pooling
  attention's concatenated value output, the pooling value vectors, and
  cosine locality between the projected probe and patch keys **within each head**.
  Locality is min-max normalized over patches independently per head and paired
  with that head's gradient/value channels. Signed head contributions are summed
  before a single ReLU. Constant-locality heads contribute zero with a warning.
  The earlier v2 implementation instead computed one locality after concatenating
  heads; it should not be mixed with these results.
- GAME propagates positive gradient-weighted pooling attention through the
  selected backbone attention rollout; it does not average all output rows.
- MaskCLIP uses the pooling values, its output projection and residual MLP as
  a dense value-only branch, weighted by probe/key cosine locality.
- GradCAM keeps the final backbone block input as its spatial feature layer.
- RISE weights masks with `sigmoid(native_logit)`, not signed logits with an
  additive bias. Gradient methods use cosine similarity as in BLIP/CLIP, avoiding
  sigmoid saturation. RISE therefore uses SigLIP's native probability target,
  unlike the existing BLIP cosine-based RISE; record this difference in comparisons.

These are architecture-specific adaptations, not a claim that the original
CLS-based algorithms transfer unchanged or guarantee better localization.
For pretrained-model comparisons, use BLIP's `--last-heads native`; the BLIP
single-head notebook adaptation intentionally changes its model forward.
Keep prompts, data, image size, attack parameters, RISE masks and layer-selection
policies matched across models. This update leaves BLIP/CLIP behavior unchanged.

Use a **new output directory** when rerunning attacks with v3: the existing
file-based resume check can otherwise skip samples produced by the old
implementation. The explainer and attack JSONs record `explainer_version`.
The shared metric still uses softmax over class logits; that is separate from
RISE's native sigmoid target.

If Grad-ECLIP becomes constant, the runner logs the cosine, pooling gradient
norm, query/key/value norms, locality range and signed contribution range.
The attack stops rather than using an arbitrary ranking from an all-zero map.
The standalone explainer still saves the map for inspection. This diagnostic
does not replace ReLU with absolute values or fabricate a nonzero explanation.

Run the CPU contract tests (random tiny models, no checkpoint downloads) with:

```bash
python -m unittest discover -s tests -p test_siglip_attack.py
```

They check native pooling outputs/gradients, pooling attention, the Grad-ECLIP
and GAME equations, RISE probability weighting, finite maps, and preservation of
the attack gradient graph. They do not measure localization on real images.

Both classification and adversarial scoring default to the same zero-shot
prompt template used by the existing CLIP pipeline: `a photo of {}`. Override
it with `--prompt-template` if needed. For the explainer smoke script, `--text`
is passed as provided.

## Adversarial scoring

`adv_score_SIGLIP.py` uses the repository's existing adversarial causal metric
engines. It accepts a sample JSON mapping sample IDs to image paths relative
to `--img-dir`, matching the BLIP runner format:

```json
{
  "sample_001": "n01518878/example.JPEG",
  "sample_002": "n02099601/example.JPEG"
}
```

Run a small test directly:

```bash
python adv_score_SIGLIP.py \
  --model google/siglip-base-patch16-224 \
  --img-dir /path/to/ImageNet/val \
  --sample-path /path/to/samples.json \
  --output-dir /path/to/SIGLIP_result \
  --hm-type eclip \
  --mode del \
  --eps 4 --alpha 1 --pgd-steps 2 \
  --process-batch-size 8
```

First create a matching sample list with the SigLIP classifier:

```bash
python classify.py \
  --model siglip \
  --siglip-model google/siglip-base-patch16-224 \
  --data-path /path/to/ImageNet/val \
  --index-json imgnet1k_label.json \
  --max-correct-samples 100 \
  --batch-size 32 \
  --prompt-template "a photo of {}" \
  --output classification_reuslt/imagenet/siglip_100_summary.json \
  --sample-output classification_reuslt/imagenet/siglip_100.json \
  --device cuda
```

`script_sh/classify_siglip.sh` provides the equivalent Slurm launcher. The
resulting `siglip_100.json` can be passed directly to the adversarial runner.

The runner supports `eclip`/`gradeclip`, `game`, `gradcam`, `maskclip`, and
`rise`; `--mode` accepts `del`, `ins`, and `del+ins`. The runner maps
CLIP-normalized tensors expected by the shared attack engines to the mean and
standard deviation in the selected SigLIP image processor. The classifier uses
SigLIP's scaled image-text similarity plus its learned logit bias, then applies
the existing softmax-based metric contract.

It skips a sample only if its adversarial image, both deletion/insertion
saliency triplets, and both summary plots are present. With `--save-process`,
it also checks all expected step previews, so incomplete samples are run again.
`curve_information.json` is written atomically after a sample finishes.

For the Slurm launchers, set `CONDA_ENV` to an environment with the
SigLIP-compatible Transformers dependency. `script_sh/test_siglip.sh` requires
`IMAGE_PATH` and `TEXT`; `script_sh/run_siglip.sh` requires `IMG_DIR` and
`SAMPLE_PATH`. The attack launcher accepts `MODEL`, `MODE`, `OUTPUT_DIR`,
`HM_TYPES`, `EPSILONS`, and the corresponding batch/PGD environment variables.
