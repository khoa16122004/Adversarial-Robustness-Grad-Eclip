import argparse
import json
import os

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize

from clip_utils import build_zero_shot_classifier
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from util import ZeroShotClipClassifier, build_blur_substrate, generate_hm, predict_zero_shot_clip
from RISE.evaluation import CausalMetric, auc, gkern


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype deletion/insertion evaluation for CLIP zero-shot explanations"
    )
    parser.add_argument("--image-path", required=True, help="Path to one RGB image")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument(
        "--hm-type",
        default="eclip",
        choices=["selfattn", "gradcam", "maskclip", "eclip", "eclip-wo-ksim", "game", "rollout", "surgery", "m2ib", "rise"],
        help="Explanation method passed to CLIPExplainRunner.generate_hm",
    )
    parser.add_argument(
        "--target-source",
        default="pred",
        choices=["pred", "gt"],
        help="Use predicted label or ground-truth label prompt for the saliency map",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="Optional ImageNet class index for GT prompt")
    parser.add_argument("--step", type=int, default=224, help="Pixels modified per causal-metric step")
    parser.add_argument("--kernel-size", type=int, default=11, help="Gaussian blur kernel size for insertion")
    parser.add_argument("--kernel-sigma", type=int, default=5, help="Gaussian blur sigma for insertion")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--output-json", default="test_eval_result.json", help="Where to save numeric results as JSON")
    parser.add_argument("--output-txt", default="test_eval_result.txt", help="Where to save a short text summary")
    return parser.parse_args()


def resolve_target_label(args, pred_label):
    if args.target_source == "pred":
        return pred_label
    if args.gt_label is None:
        raise ValueError("--gt-label is required when --target-source gt")
    if not (0 <= args.gt_label < len(IMAGENET_CLASSNAMES)):
        raise ValueError("--gt-label must be a valid ImageNet class index")
    return int(args.gt_label)


def save_outputs(output_json, output_txt, payload):
    output_json_dir = os.path.dirname(output_json)
    output_txt_dir = os.path.dirname(output_txt)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"image_path: {payload['image_path']}",
        f"clip_model: {payload['clip_model']}",
        f"hm_type: {payload['hm_type']}",
        f"pred_label: {payload['pred_label']}",
        f"pred_classname: {payload['pred_classname']}",
        f"pred_confidence: {payload['pred_confidence']:.6f}",
        f"target_source: {payload['target_source']}",
        f"target_label: {payload['target_label']}",
        f"target_classname: {payload['target_classname']}",
        f"deletion_auc: {payload['deletion_auc']:.6f}",
        f"insertion_auc: {payload['insertion_auc']:.6f}",
    ]
    if payload["gt_label"] is not None:
        lines.append(f"gt_label: {payload['gt_label']}")
        lines.append(f"gt_classname: {payload['gt_classname']}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    zero_shot_weights = build_zero_shot_classifier(
        clip_model,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )
    classifier = ZeroShotClipClassifier(clip_model=clip_model, zero_shot_weights=zero_shot_weights)
    classifier.eval()

    image = Image.open(args.image_path).convert("RGB")
    input_resolution = clip_model.visual.input_resolution
    resized_image = image.resize((input_resolution, input_resolution), Image.BICUBIC)
    image_tensor = preprocess(resized_image).unsqueeze(0)
    metric_resize = Resize(tuple(image_tensor.shape[-2:]))

    _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_tensor, device)

    target_label = resolve_target_label(args, pred_label)
    target_texts = [IMAGENET_CLASSNAMES[target_label]]

    with torch.no_grad():
        text_tokens = clip.tokenize(target_texts).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, dim=-1)

    heatmap = generate_hm(
        clip_model,
        args.hm_type,
        resized_image,
        text_embedding,
        target_texts,
        metric_resize,
        preprocess,
    )
    saliency = heatmap.detach().cpu().numpy()

    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    insertion = CausalMetric(classifier, "ins", args.step, substrate_fn=blur_fn)
    deletion = CausalMetric(classifier, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))

    deletion_curve = deletion.single_run(image_tensor, saliency, verbose=0)
    insertion_curve = insertion.single_run(image_tensor, saliency, verbose=0)

    gt_classname = IMAGENET_CLASSNAMES[args.gt_label] if args.gt_label is not None else None
    payload = {
        "image_path": os.path.abspath(args.image_path),
        "clip_model": args.clip_model,
        "device": device,
        "hm_type": args.hm_type,
        "target_source": args.target_source,
        "gt_label": args.gt_label,
        "gt_classname": gt_classname,
        "pred_label": pred_label,
        "pred_classname": IMAGENET_CLASSNAMES[pred_label],
        "pred_confidence": pred_confidence,
        "target_label": target_label,
        "target_classname": IMAGENET_CLASSNAMES[target_label],
        "step": args.step,
        "kernel_size": args.kernel_size,
        "kernel_sigma": args.kernel_sigma,
        "deletion_auc": float(auc(deletion_curve)),
        "insertion_auc": float(auc(insertion_curve)),
        "deletion_curve": deletion_curve.tolist(),
        "insertion_curve": insertion_curve.tolist(),
    }

    save_outputs(args.output_json, args.output_txt, payload)

    print(
        json.dumps(
            {
                "pred_label": payload["pred_label"],
                "pred_classname": payload["pred_classname"],
                "target_label": payload["target_label"],
                "target_classname": payload["target_classname"],
                "deletion_auc": payload["deletion_auc"],
                "insertion_auc": payload["insertion_auc"],
                "output_json": os.path.abspath(args.output_json),
                "output_txt": os.path.abspath(args.output_txt),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
