import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from CLIP import clip
except ImportError:
    import clip

import torch

from clip_utils import build_zero_shot_classifier
from util import (
    DEFAULT_PROMPT_TEMPLATE,
    batched,
    build_classnames_from_label_map,
    load_classnames,
    load_dataset_label_map,
)


WNID_PATTERN = re.compile(r"^n\d{8}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class AttackImageItem:
    image_path: str
    rel_path: str
    attack_root: str
    attack_tag: str
    wnid: str
    gt_label: int


def get_normalize_stats(preprocess):
    transforms = getattr(preprocess, "transforms", [])
    for transform in transforms:
        if hasattr(transform, "mean") and hasattr(transform, "std"):
            mean = torch.tensor(transform.mean, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(transform.std, dtype=torch.float32).view(3, 1, 1)
            return mean, std
    raise ValueError("Could not infer normalize mean/std from CLIP preprocess")


def preprocess_adv_image(image: Image.Image, mean: torch.Tensor, std: torch.Tensor, expected_size: int):
    # For adversarial images: keep original spatial layout, only convert to tensor and normalize.
    rgb = image.convert("RGB")
    np_img = np.asarray(rgb, dtype=np.float32)
    if np_img.ndim != 3 or np_img.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={np_img.shape}")

    height, width = int(np_img.shape[0]), int(np_img.shape[1])
    if height != expected_size or width != expected_size:
        raise ValueError(
            f"Unexpected adv image size {width}x{height}. Expected {expected_size}x{expected_size}. "
            "This script disables spatial transforms by design."
        )

    tensor = torch.from_numpy(np_img).permute(2, 0, 1) / 255.0
    return (tensor - mean) / std


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run CLIP zero-shot inference on adversarial images and check retain-class "
            "(pred == original class) for all samples."
        )
    )
    parser.add_argument(
        "--attack-roots",
        nargs="+",
        required=True,
        help=(
            "One or more folders that contain attack results. "
            "Example: D:/neurips_result/result/ImageNet/CLIP_ViTB16/Xshift_score"
        ),
    )
    parser.add_argument(
        "--index-json",
        default="imgnet1k_label.json",
        help="Label mapping json that contains wnid <-> class index information.",
    )
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name for clip.load")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument(
        "--classnames-path",
        default=None,
        help="Optional path to class names (.txt or .json). Default: infer from --index-json, else ImageNet.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template for zero-shot text, use {} as class placeholder",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for image inference")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on evaluated images")
    parser.add_argument(
        "--filename-contains",
        default="adversarial_image",
        help="Only evaluate image files whose filename contains this string (case-insensitive)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument(
        "--output",
        default="reinference_attack_retain_report.json",
        help="Output JSON path for full report",
    )
    return parser.parse_args()


def infer_wnid_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        if WNID_PATTERN.match(part):
            return part
    return None


def infer_attack_tag(path: Path) -> str:
    parts = list(path.parts)
    for part in parts:
        if "score" in part.lower():
            return part
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def collect_attack_images(
    attack_roots: Sequence[str],
    folder_to_label: Dict[str, int],
    filename_contains: str,
    max_images: Optional[int],
) -> Tuple[List[AttackImageItem], List[Dict[str, str]]]:
    selected: List[AttackImageItem] = []
    skipped: List[Dict[str, str]] = []
    needle = filename_contains.lower()

    for attack_root in attack_roots:
        root_path = Path(attack_root)
        if not root_path.exists():
            skipped.append({"path": str(root_path), "reason": "attack_root_not_found"})
            continue

        for dirpath, _, filenames in os.walk(root_path):
            dir_path = Path(dirpath)
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                if suffix not in IMAGE_EXTENSIONS:
                    continue
                if needle and needle not in filename.lower():
                    continue

                image_path = dir_path / filename
                wnid = infer_wnid_from_path(image_path)
                if wnid is None:
                    skipped.append({"path": str(image_path), "reason": "wnid_not_found_in_path"})
                    continue

                if wnid not in folder_to_label:
                    skipped.append({"path": str(image_path), "reason": f"wnid_not_in_label_map:{wnid}"})
                    continue

                selected.append(
                    AttackImageItem(
                        image_path=str(image_path),
                        rel_path=str(image_path.relative_to(root_path)).replace("\\", "/"),
                        attack_root=str(root_path),
                        attack_tag=infer_attack_tag(image_path),
                        wnid=wnid,
                        gt_label=int(folder_to_label[wnid]),
                    )
                )

                if max_images is not None and len(selected) >= max_images:
                    return selected, skipped

    return selected, skipped


def ensure_classnames(args, label_to_classname: Dict[int, str]) -> List[str]:
    if args.classnames_path:
        return load_classnames(args.classnames_path)

    classnames = build_classnames_from_label_map(label_to_classname)
    if classnames:
        return classnames

    return load_classnames(None)


def summarize_by_attack(items: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(lambda: {"total": 0, "retain": 0})
    for item in items:
        key = item["attack_tag"]
        grouped[key]["total"] += 1
        grouped[key]["retain"] += int(bool(item["retain_class"]))

    summary = {}
    for key, stats in grouped.items():
        total = int(stats["total"])
        retain = int(stats["retain"])
        summary[key] = {
            "total": total,
            "retain": retain,
            "retain_rate": (float(retain) / float(total)) if total > 0 else 0.0,
        }
    return summary


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    folder_to_label, label_to_classname = load_dataset_label_map(args.index_json)
    classnames = ensure_classnames(args, label_to_classname)

    max_gt_label = max(folder_to_label.values())
    if max_gt_label >= len(classnames):
        raise ValueError(
            f"Class-name count ({len(classnames)}) is smaller than max label index ({max_gt_label})."
        )

    image_items, skipped_before_infer = collect_attack_images(
        attack_roots=args.attack_roots,
        folder_to_label=folder_to_label,
        filename_contains=args.filename_contains,
        max_images=args.max_images,
    )

    if not image_items:
        raise ValueError(
            "No adversarial image found. Check --attack-roots and --filename-contains. "
            "Default filename filter is 'adversarial_image'."
        )

    clip_source = args.clip_checkpoint or args.clip_model
    if args.clip_checkpoint and not os.path.isfile(args.clip_checkpoint):
        raise FileNotFoundError(f"--clip-checkpoint not found: {args.clip_checkpoint}")

    clip_model, preprocess = clip.load(clip_source, device=device)
    clip_model.eval()
    input_resolution = int(clip_model.visual.input_resolution)
    norm_mean, norm_std = get_normalize_stats(preprocess)

    zero_shot_weights = build_zero_shot_classifier(
        clip_model,
        classnames=classnames,
        templates=(args.prompt_template,),
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    results: List[Dict] = []
    skipped_runtime: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch in tqdm(list(batched(image_items, args.batch_size)), desc="Re-inference"):
            tensors = []
            metas: List[AttackImageItem] = []

            for item in batch:
                try:
                    image = Image.open(item.image_path)
                except Exception as exc:  # pylint: disable=broad-except
                    skipped_runtime.append({"path": item.image_path, "reason": f"open_error:{exc}"})
                    continue

                try:
                    image_tensor = preprocess_adv_image(image, norm_mean, norm_std, input_resolution)
                except Exception as exc:  # pylint: disable=broad-except
                    skipped_runtime.append({"path": item.image_path, "reason": f"preprocess_error:{exc}"})
                    continue

                tensors.append(image_tensor)
                metas.append(item)

            if not tensors:
                continue

            image_batch = torch.stack(tensors, dim=0).to(device)
            image_features = clip_model.encode_image(image_batch)
            logits = 100.0 * image_features @ zero_shot_weights
            probs = logits.softmax(dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for i, item in enumerate(metas):
                pred_label = int(preds[i].item())
                confidence = float(probs[i, pred_label].item())
                retain_class = pred_label == item.gt_label

                results.append(
                    {
                        "image_path": item.image_path,
                        "attack_root": item.attack_root,
                        "relative_path": item.rel_path,
                        "attack_tag": item.attack_tag,
                        "wnid": item.wnid,
                        "gt_label": item.gt_label,
                        "gt_classname": classnames[item.gt_label],
                        "pred_label": pred_label,
                        "pred_classname": classnames[pred_label],
                        "confidence": confidence,
                        "retain_class": retain_class,
                    }
                )

    total_eval = len(results)
    total_retain = sum(int(x["retain_class"]) for x in results)
    retain_rate = (float(total_retain) / float(total_eval)) if total_eval > 0 else 0.0

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model": f"CLIP-{args.clip_model}",
        "clip_source": args.clip_checkpoint or args.clip_model,
        "device": device,
        "attack_roots": [str(Path(x)) for x in args.attack_roots],
        "index_json": args.index_json,
        "classnames_path": args.classnames_path,
        "prompt_template": args.prompt_template,
        "batch_size": args.batch_size,
        "filename_contains": args.filename_contains,
        "max_images": args.max_images,
        "summary": {
            "total_found": len(image_items),
            "total_evaluated": total_eval,
            "total_retain": total_retain,
            "retain_rate": retain_rate,
            "skipped_before_infer": len(skipped_before_infer),
            "skipped_runtime": len(skipped_runtime),
        },
        "by_attack": summarize_by_attack(results),
        "skipped_before_infer": skipped_before_infer,
        "skipped_runtime": skipped_runtime,
        "samples": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {args.output}")
    print(f"Found images: {len(image_items)}")
    print(f"Evaluated: {total_eval}")
    print(f"Retain class: {total_retain}")
    print(f"Retain rate: {retain_rate:.6f}")


if __name__ == "__main__":
    main()
