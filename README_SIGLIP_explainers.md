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
  --rise-batch-size 50
```

The script checks its custom image encoder against `SiglipModel.get_image_features`
before generating the maps. It saves each method's NumPy map, heatmap, and
overlay, plus `summary.json`. SigLIP has no class token in its vision patch
sequence, so GradeCAM-style locality and GAME relevance aggregate over the
patch tokens instead of selecting a CLS token.

The default zero-shot text template follows the Transformers documentation:
`This is a photo of {}.` For the explainer smoke script, `--text` is passed as
provided.

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
