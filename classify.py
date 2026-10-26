import argparse
import json
import os
import random
from pathlib import Path

try:
    from CLIP import clip
except ImportError:
    import clip
import torch
from PIL import Image
from tqdm import tqdm

from clip_utils import build_zero_shot_classifier
from util import (
    DEFAULT_PROMPT_TEMPLATE,
    batched,
    build_classnames_from_label_map,
    collect_image_items,
    load_dataset_label_map,
    load_classnames,
)


def main():
    parser = argparse.ArgumentParser(
        description="Classify ImageNet-style samples with CLIP or BLIP zero-shot"
    )
    parser.add_argument("--data-path", required=True, help="Path to dataset folder (class subfolders)")
    parser.add_argument(
        "--index-json",
        default="imgnet1k_label.json",
        help=(
            "Path to label mapping json. Supports ImageNet style and generic formats with "
            "folder/class_index/class_name"
        ),
    )
    parser.add_argument("--model", choices=("clip", "blip"), default="clip")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name for clip.load")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument("--max-correct-samples", type=int, default=None,
                        help="Randomly select up to this many correctly classified samples for sample output")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for randomized BLIP sample selection",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for image inference")
    parser.add_argument(
        "--classnames-path",
        default=None,
        help="Optional path to class names (.txt or .json). Default: ImageNet class names",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template for zero-shot text, use {} as class placeholder",
    )
    parser.add_argument(
        "--output",
        default="D:/Adversarial-Robustness-Grad-Eclip/densenet121.json",
        help="Classification summary JSON path",
    )
    parser.add_argument(
        "--sample-output",
        default=None,
        help="Optional attack-ready JSON path mapping sample IDs to relative image paths",
    )
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--blip-root", type=Path, default=repo_root / "BLIP",
                        help="BLIP source directory (used with --model blip)")
    parser.add_argument("--blip-checkpoint", type=Path,
                        default=repo_root / "blip_checkpoint.pth",
                        help="BLIP checkpoint path (used with --model blip)")
    parser.add_argument("--vit", choices=("base", "large"), default="base",
                        help="BLIP visual encoder size")
    parser.add_argument("--blip-last-heads", choices=("1", "native"), default="native",
                        help="BLIP final attention mode (use native for pretrained classification)")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    if args.max_correct_samples is not None and args.max_correct_samples < 1:
        raise ValueError("--max-correct-samples must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    folder_to_label, label_to_classname = load_dataset_label_map(args.index_json)
    image_items = collect_image_items(args.data_path, folder_to_label, max_images=args.max_images)
    if not image_items:
        raise ValueError("No image found for classification.")

    if args.classnames_path:
        classnames = load_classnames(args.classnames_path)
    else:
        classnames = build_classnames_from_label_map(label_to_classname)
        if not classnames:
            raise ValueError(
                "Cannot infer class names from --index-json. "
                "Please provide --classnames-path or include class_name in mapping json."
            )

    if args.model == "blip":
        classnames = [name.replace("_", " ") for name in classnames]

    max_gt_label = max(folder_to_label.values())
    if max_gt_label >= len(classnames):
        raise ValueError(
            f"Class-name count ({len(classnames)}) is smaller than max label index ({max_gt_label})."
        )

    if args.model == "clip":
        clip_source = args.clip_checkpoint or args.clip_model
        if args.clip_checkpoint and not os.path.isfile(args.clip_checkpoint):
            raise FileNotFoundError(f"--clip-checkpoint not found: {args.clip_checkpoint}")

        clip_model, preprocess = clip.load(clip_source, device=device)
        clip_model.eval()
        zero_shot_weights = build_zero_shot_classifier(
            clip_model,
            classnames=classnames,
            templates=(args.prompt_template,),
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )
        classifier = None
        description = "Classifying with CLIP"
    else:
        if not args.blip_checkpoint.is_file():
            raise FileNotFoundError(f"--blip-checkpoint not found: {args.blip_checkpoint}")
        if not (args.blip_root / "models" / "blip_itm.py").is_file():
            raise FileNotFoundError(
                f"--blip-root must contain models\\blip_itm.py: {args.blip_root}"
            )

        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(args.blip_root.resolve()))
        from blip_attack_adapter import BLIPAttackAdapter, build_zero_shot_blip_classifier

        adapter_args = SimpleNamespace(
            blip_root=args.blip_root,
            checkpoint=args.blip_checkpoint,
            image_size=224,
            vit=args.vit,
            last_heads=args.blip_last_heads,
            game_start_layer=-1,
            rise_masks=2000,
            rise_grid=8,
            rise_prob=0.5,
            rise_batch_size=50,
            seed=42,
        )
        adapter = BLIPAttackAdapter(adapter_args, device)
        preprocess = adapter.preprocess
        classifier = build_zero_shot_blip_classifier(
            adapter, classnames, args.prompt_template,
        )
        zero_shot_weights = None
        clip_model = None
        description = "Classifying with BLIP"

    correct_samples = []
    num_total = 0
    num_correct = 0

    with torch.no_grad():
        for batch in tqdm(batched(image_items, args.batch_size), desc=description):
            tensors = []
            metas = []
            for image_path, rel_path, wnid, gt_label in batch:
                image = Image.open(image_path).convert("RGB")
                tensors.append(preprocess(image))
                metas.append((rel_path, wnid, gt_label))

            image_batch = torch.stack(tensors, dim=0).to(device)
            if args.model == "clip":
                image_features = clip_model.encode_image(image_batch)
                logits = 100.0 * image_features @ zero_shot_weights
            else:
                logits = classifier(image_batch)
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
        "model": f"CLIP-{args.clip_model}" if args.model == "clip" else f"BLIP-ViT-{args.vit}",
        "device": device,
        "data_path": args.data_path,
        "index_json": args.index_json,
        "num_classes": len(classnames),
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "num_total_processed": num_total,
        "num_correct": num_correct,
        "accuracy": (float(num_correct) / float(num_total)) if num_total > 0 else 0.0,
        "correct_samples": correct_samples,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.sample_output:
        sample_output = Path(args.sample_output)
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        samples_to_write = correct_samples
        if args.max_correct_samples is not None:
            rng = random.Random(args.seed)
            samples_by_wnid = {}
            shuffled_correct_samples = correct_samples.copy()
            rng.shuffle(shuffled_correct_samples)
            for sample in shuffled_correct_samples:
                samples_by_wnid.setdefault(sample["wnid"], sample)
            class_samples = list(samples_by_wnid.values())
            samples_to_write = rng.sample(
                class_samples, min(args.max_correct_samples, len(class_samples))
            )

        sample_payload = {}
        for sample in samples_to_write:
            sample_id = sample["wnid"]
            suffix = 2
            while sample_id in sample_payload:
                sample_id = f"{sample['wnid']}_{suffix}"
                suffix += 1
            sample_payload[sample_id] = sample["image_path"]
        with sample_output.open("w", encoding="utf-8") as f:
            json.dump(sample_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved attack samples: {sample_output}")

    print(f"Saved: {args.output}")
    print(f"Processed: {num_total}")
    print(f"Correct: {num_correct}")


if __name__ == "__main__":
    main()