import argparse
import json
import os

import clip
import torch
from PIL import Image
from tqdm import tqdm

from clip_utils import build_zero_shot_classifier
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from util import batched, collect_image_items, load_imagenet_label_map


def main():
    parser = argparse.ArgumentParser(
        description="Classify ImageNet-style samples with CLIP zero-shot and save correct samples to JSON"
    )
    parser.add_argument("--data-path", required=True, help="Path to ImageNet-style val folder (wnid subfolders)")
    parser.add_argument("--index-json", default="imgnet1k_label.json", help="Path to ImageNet class index json")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name for clip.load")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CLIP image inference")
    parser.add_argument(
        "--output",
        default="D:/Adversarial-Robustness-Grad-Eclip/densenet121.json",
        help="Output JSON path",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    folder_to_label = load_imagenet_label_map(args.index_json)
    image_items = collect_image_items(args.data_path, folder_to_label, max_images=args.max_images)
    if not image_items:
        raise ValueError("No image found for classification.")

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

    correct_samples = []
    num_total = 0
    num_correct = 0

    with torch.no_grad():
        for batch in tqdm(list(batched(image_items, args.batch_size)), desc="Classifying with CLIP"):
            tensors = []
            metas = []
            for image_path, rel_path, wnid, gt_label in batch:
                image = Image.open(image_path).convert("RGB")
                tensors.append(preprocess(image))
                metas.append((rel_path, wnid, gt_label))

            image_batch = torch.stack(tensors, dim=0).to(device)
            image_features = clip_model.encode_image(image_batch)
            logits = 100.0 * image_features @ zero_shot_weights
            probs = logits.softmax(dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for i, (rel_path, wnid, gt_label) in enumerate(metas):
                pred_label = int(preds[i].item())
                confidence = float(probs[i, pred_label].item())
                num_total += 1

                if pred_label == gt_label:
                    num_correct += 1
                    correct_samples.append(
                        {
                            "image_path": rel_path,
                            "wnid": wnid,
                            "gt_label": int(gt_label),
                            "pred_label": pred_label,
                            "confidence": confidence,
                        }
                    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model": f"CLIP-{args.clip_model}",
        "device": device,
        "data_path": args.data_path,
        "index_json": args.index_json,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "num_total_processed": num_total,
        "num_correct": num_correct,
        "accuracy": (float(num_correct) / float(num_total)) if num_total > 0 else 0.0,
        "correct_samples": correct_samples,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output}")
    print(f"Processed: {num_total}")
    print(f"Correct: {num_correct}")


if __name__ == "__main__":
    main()
